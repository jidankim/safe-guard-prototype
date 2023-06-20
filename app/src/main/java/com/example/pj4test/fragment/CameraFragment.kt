/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.pj4test.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.pj4test.ProjectConfiguration
import com.example.pj4test.audioInference.SnapClassifier
import java.util.LinkedList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.example.pj4test.cameraInference.PersonClassifier
import com.example.pj4test.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.File
import java.time.LocalDateTime
import java.util.Date

class CameraFragment : Fragment(), PersonClassifier.DetectorListener {
    private val TAG = "CameraFragment"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!
    
    private lateinit var personView: TextView
    
    private lateinit var personClassifier: PersonClassifier
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var snapClassifier: SnapClassifier
    // Camera and audio recording
    private lateinit var mediaRecorder: MediaRecorder
    private var isRecording: Boolean = false
    private var lastRecordedTime: LocalDateTime? = null

    // Email settings
    private val senderEmail = "your_email@example.com"
    private val senderPassword = "your_email_password"
    private val recipientEmail = "recipient_email@example.com"

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)

        // Create an instance of SnapClassifier
        snapClassifier = SnapClassifier()

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        personClassifier = PersonClassifier()
        personClassifier.initialize(requireContext())
        personClassifier.setDetectorListener(this)

        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        personView = fragmentCameraBinding.PersonView

        // Create an instance of SnapClassifier
        snapClassifier = SnapClassifier()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                val cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases(cameraProvider)

//                snapClassifier.setObjectDetectionListener(this)
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases(cameraProvider: ProcessCameraProvider) {

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()
        // Attach the viewfinder's surface provider to preview use case
        preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)


        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
        // The analyzer can then be assigned to the instance
        imageAnalyzer!!.setAnalyzer(cameraExecutor) { image -> detectObjects(image) }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    private fun detectObjects(image: ImageProxy) {
        if (!::bitmapBuffer.isInitialized) {
            // The image rotation and RGB image buffer are initialized only once
            // the analyzer has started running
            bitmapBuffer = Bitmap.createBitmap(
                image.width,
                image.height,
                Bitmap.Config.ARGB_8888
            )
        }
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees

        // Pass Bitmap and rotation to the object detector helper for processing and detection
        personClassifier.detect(bitmapBuffer, imageRotation)
    }

    // Update UI after objects have been detected. Extracts original image height/width
    // to scale and place bounding boxes properly through OverlayView
    @RequiresApi(Build.VERSION_CODES.S)
    override fun onObjectDetectionResults(
        results: MutableList<Detection>?,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        activity?.runOnUiThread {
            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )
            
            // find at least one bounding box of the person
            val isPersonDetected: Boolean = results!!.find { it.categories[0].label == "person" } != null
            
            // change UI according to the result
            if (isPersonDetected) {
                personView.text = "UNAUTHORIZED PERSON"
                personView.setBackgroundColor(ProjectConfiguration.activeBackgroundColor)
                personView.setTextColor(ProjectConfiguration.activeTextColor)

                // Turn off camera and audio recording
                if (isRecording && (lastRecordedTime?.isBefore(LocalDateTime.now().minusSeconds(10)) == true)) {
                    Log.d(TAG, "Stopping Recording with lastRecordedTime $lastRecordedTime")
                    stopRecording()
                    // sendRecordedDataViaEmail()
                }
            } else {
                personView.text = "AUTHORIZED PERSON"
                personView.setBackgroundColor(ProjectConfiguration.idleBackgroundColor)
                personView.setTextColor(ProjectConfiguration.idleTextColor)

                // Turn on camera and audio recording
                if (!isRecording) {
                    // Call the startRecording() method of the SnapClassifier instance
                    startRecording()
                }
            }

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onObjectDetectionError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun startRecording() {
        try {
            snapClassifier.startRecording()
            snapClassifier.startInferencing()
            // Set up the media recorder
//            val outputFile = createOutputFile()
//            mediaRecorder = MediaRecorder(requireContext()).apply {
//                setAudioSource(MediaRecorder.AudioSource.DEFAULT)
//                setOutputFormat(MediaRecorder.OutputFormat.DEFAULT)
//                setAudioEncoder(MediaRecorder.AudioEncoder.DEFAULT)
//                setOutputFile(outputFile.absolutePath)
//                prepare()
//                start()
//            }
            isRecording = true
            lastRecordedTime = LocalDateTime.now()
            Log.d(TAG, "While starting Recording last recorded time to $lastRecordedTime from null")
            Log.d(TAG, "Recording started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording: ${e.message}")
        }
    }

    private fun stopRecording() {
        try {
//            mediaRecorder.stop()
//            mediaRecorder.release()
            snapClassifier.stopInferencing()
            snapClassifier.stopRecording()
            isRecording = false
            Log.d(TAG, "While ending Recording last recorded time to null from $lastRecordedTime")
            lastRecordedTime = null
            Log.d(TAG, "Recording ended")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop recording: ${e.message}")
        }
    }

    private fun createOutputFile(): File {
        val outputDir = requireContext().getExternalFilesDir(null)
        val timestamp = Date().time
        return File.createTempFile("recording_${timestamp}_", ".3gp", outputDir)
    }
}

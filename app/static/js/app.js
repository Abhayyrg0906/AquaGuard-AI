/**
 * AquaGuard AI - Frontend Controller
 * Handles tabs, file uploads, sample selection, and REST API interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // State Management
    // -------------------------------------------------------------------------
    const state = {
        imageFile: null,
        selectedImageSampleId: null,
        videoFile: null,
        selectedVideoSampleId: null,
        samples: []
    };

    // -------------------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------------------
    // Status
    const statusDot = document.getElementById('system-status-dot');
    const statusText = document.getElementById('system-status-text');
    const headerModelBadge = document.getElementById('header-model-badge');

    // Navigation
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Image Tab Elements
    const imageDropzone = document.getElementById('image-dropzone');
    const imageFileInput = document.getElementById('image-file-input');
    const imageFileInfo = document.getElementById('image-file-info');
    const imageFileName = document.getElementById('image-file-name');
    const imageClearBtn = document.getElementById('image-clear-btn');
    const imageSampleChips = document.getElementById('image-sample-chips');
    const imageConfSlider = document.getElementById('image-conf-slider');
    const imageConfVal = document.getElementById('image-conf-val');
    const imageIouSlider = document.getElementById('image-iou-slider');
    const imageIouVal = document.getElementById('image-iou-val');
    const btnRunImage = document.getElementById('btn-run-image');
    const imageSpinner = document.getElementById('image-spinner');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const imageOutputContainer = document.getElementById('image-output-container');
    const imageAlertBanner = document.getElementById('image-alert-banner');
    const statDetectionCount = document.getElementById('stat-detection-count');
    const statInferenceTime = document.getElementById('stat-inference-time');
    const statImageDims = document.getElementById('stat-image-dims');
    const annotatedImageDisplay = document.getElementById('annotated-image-display');
    const detectionsTableBody = document.getElementById('detections-table-body');

    // Video Tab Elements
    const videoDropzone = document.getElementById('video-dropzone');
    const videoFileInput = document.getElementById('video-file-input');
    const videoFileInfo = document.getElementById('video-file-info');
    const videoFileName = document.getElementById('video-file-name');
    const videoClearBtn = document.getElementById('video-clear-btn');
    const videoSampleChips = document.getElementById('video-sample-chips');
    const videoConfSlider = document.getElementById('video-conf-slider');
    const videoConfVal = document.getElementById('video-conf-val');
    const videoIouSlider = document.getElementById('video-iou-slider');
    const videoIouVal = document.getElementById('video-iou-val');
    const btnRunVideo = document.getElementById('btn-run-video');
    const videoSpinner = document.getElementById('video-spinner');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const videoOutputContainer = document.getElementById('video-output-container');
    const videoAlertBanner = document.getElementById('video-alert-banner');
    const statVideoFrames = document.getElementById('stat-video-frames');
    const statVideoDetections = document.getElementById('stat-video-detections');
    const statVideoFps = document.getElementById('stat-video-fps');
    const statVideoLatency = document.getElementById('stat-video-latency');
    const annotatedVideoPlayer = document.getElementById('annotated-video-player');

    // -------------------------------------------------------------------------
    // Tab Navigation
    // -------------------------------------------------------------------------
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            navTabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // -------------------------------------------------------------------------
    // Sliders Real-time Binding
    // -------------------------------------------------------------------------
    imageConfSlider.addEventListener('input', (e) => imageConfVal.textContent = parseFloat(e.target.value).toFixed(2));
    imageIouSlider.addEventListener('input', (e) => imageIouVal.textContent = parseFloat(e.target.value).toFixed(2));
    videoConfSlider.addEventListener('input', (e) => videoConfVal.textContent = parseFloat(e.target.value).toFixed(2));
    videoIouSlider.addEventListener('input', (e) => videoIouVal.textContent = parseFloat(e.target.value).toFixed(2));

    // -------------------------------------------------------------------------
    // System Health & Model Info Initialization
    // -------------------------------------------------------------------------
    async function initSystem() {
        try {
            const healthRes = await fetch('/api/v1/health');
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                if (healthData.model_loaded) {
                    statusDot.className = 'status-dot active';
                    statusText.textContent = `Model Online (${healthData.device})`;
                    headerModelBadge.textContent = `YOLOv8n • ${healthData.device}`;
                } else {
                    statusDot.className = 'status-dot error';
                    statusText.textContent = 'Model Degraded / Missing';
                }
            } else {
                statusDot.className = 'status-dot error';
                statusText.textContent = 'API Unavailable';
            }
        } catch (err) {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Offline';
        }

        // Fetch Samples
        loadSamples();
    }

    async function loadSamples() {
        try {
            const res = await fetch('/api/v1/samples');
            if (!res.ok) return;
            const data = await res.json();
            state.samples = data.samples || [];

            // Populate Image Samples
            const imageSamples = state.samples.filter(s => s.type === 'image');
            if (imageSamples.length > 0) {
                imageSampleChips.innerHTML = '';
                imageSamples.forEach((sample, idx) => {
                    const btn = document.createElement('button');
                    btn.className = 'chip';
                    btn.textContent = `Sample ${idx + 1}`;
                    btn.title = sample.description;
                    btn.dataset.sampleId = sample.id;
                    btn.addEventListener('click', () => selectImageSample(sample.id, btn));
                    imageSampleChips.appendChild(btn);
                });
            }

            // Populate Video Samples
            const videoSamples = state.samples.filter(s => s.type === 'video');
            if (videoSamples.length > 0) {
                videoSampleChips.innerHTML = '';
                videoSamples.forEach((sample) => {
                    const btn = document.createElement('button');
                    btn.className = 'chip';
                    btn.textContent = sample.name;
                    btn.title = sample.description;
                    btn.dataset.sampleId = sample.id;
                    btn.addEventListener('click', () => selectVideoSample(sample.id, btn));
                    videoSampleChips.appendChild(btn);
                });
            }
        } catch (err) {
            console.warn('Failed to load sample assets:', err);
        }
    }

    // -------------------------------------------------------------------------
    // Image Upload & Selection Handling
    // -------------------------------------------------------------------------
    imageDropzone.addEventListener('click', (e) => {
        if (!e.target.classList.contains('btn-clear')) {
            imageFileInput.click();
        }
    });

    imageDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropzone.classList.add('dragover');
    });

    imageDropzone.addEventListener('dragleave', () => {
        imageDropzone.classList.remove('dragover');
    });

    imageDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    function handleImageFile(file) {
        state.imageFile = file;
        state.selectedImageSampleId = null;
        clearImageSampleSelection();

        imageFileName.textContent = file.name;
        imageDropzone.querySelector('.dropzone-content').classList.add('hidden');
        imageFileInfo.classList.remove('hidden');
    }

    imageClearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetImageUpload();
    });

    function resetImageUpload() {
        state.imageFile = null;
        imageFileInput.value = '';
        imageFileInfo.classList.add('hidden');
        imageDropzone.querySelector('.dropzone-content').classList.remove('hidden');
    }

    function selectImageSample(sampleId, chipElement) {
        resetImageUpload();
        state.selectedImageSampleId = sampleId;

        document.querySelectorAll('#image-sample-chips .chip').forEach(c => c.classList.remove('selected'));
        chipElement.classList.add('selected');
    }

    function clearImageSampleSelection() {
        document.querySelectorAll('#image-sample-chips .chip').forEach(c => c.classList.remove('selected'));
    }

    // -------------------------------------------------------------------------
    // Run Image Detection
    // -------------------------------------------------------------------------
    btnRunImage.addEventListener('click', async () => {
        if (!state.imageFile && !state.selectedImageSampleId) {
            alert('Please upload an image file or select a sample scene first.');
            return;
        }

        const formData = new FormData();
        if (state.imageFile) {
            formData.append('file', state.imageFile);
        } else if (state.selectedImageSampleId) {
            formData.append('sample_id', state.selectedImageSampleId);
        }

        formData.append('confidence', imageConfSlider.value);
        formData.append('iou', imageIouSlider.value);

        // UI Loading State
        btnRunImage.disabled = true;
        imageSpinner.classList.remove('hidden');
        btnRunImage.querySelector('.btn-text').textContent = 'Processing Scene...';

        try {
            const res = await fetch('/api/v1/predict/image', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || 'Inference execution failed.');
            }

            // Display Results
            renderImageResults(data);

        } catch (err) {
            console.error('Image prediction error:', err);
            imageAlertBanner.className = 'alert-banner error';
            imageAlertBanner.textContent = `Error: ${err.message}`;
            imageAlertBanner.style.display = 'block';
            imagePlaceholder.classList.add('hidden');
            imageOutputContainer.classList.remove('hidden');
        } finally {
            btnRunImage.disabled = false;
            imageSpinner.classList.add('hidden');
            btnRunImage.querySelector('.btn-text').textContent = '🔍 Run Plastic Detection';
        }
    });

    function renderImageResults(data) {
        imagePlaceholder.classList.add('hidden');
        imageOutputContainer.classList.remove('hidden');

        // Update Stats
        statDetectionCount.textContent = data.detection_count;
        statInferenceTime.textContent = `${data.inference_time_ms} ms`;
        statImageDims.textContent = `${data.image_width} × ${data.image_height}`;

        // Alert Banner
        if (data.detection_count > 0) {
            imageAlertBanner.className = 'alert-banner success';
            imageAlertBanner.textContent = `🎯 Detected ${data.detection_count} floating plastic waste object(s) with high confidence.`;
        } else {
            imageAlertBanner.className = 'alert-banner warning';
            imageAlertBanner.textContent = `⚠️ No plastic waste detected at confidence threshold ≥ ${parseFloat(imageConfSlider.value).toFixed(2)}.`;
        }

        // Set Image
        annotatedImageDisplay.src = data.annotated_image_url;

        // Render Table
        detectionsTableBody.innerHTML = '';
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach((d, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><span class="badge-class">${d.class_name}</span></td>
                    <td class="confidence-cell">${(d.confidence * 100).toFixed(1)}%</td>
                    <td class="coord-cell">[${d.x1}, ${d.y1}, ${d.x2}, ${d.y2}]</td>
                `;
                detectionsTableBody.appendChild(tr);
            });
        } else {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--text-muted);">No plastic objects detected</td>`;
            detectionsTableBody.appendChild(tr);
        }
    }

    // -------------------------------------------------------------------------
    // Video Upload & Selection Handling
    // -------------------------------------------------------------------------
    videoDropzone.addEventListener('click', (e) => {
        if (!e.target.classList.contains('btn-clear')) {
            videoFileInput.click();
        }
    });

    videoDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoDropzone.classList.add('dragover');
    });

    videoDropzone.addEventListener('dragleave', () => {
        videoDropzone.classList.remove('dragover');
    });

    videoDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });

    function handleVideoFile(file) {
        state.videoFile = file;
        state.selectedVideoSampleId = null;
        clearVideoSampleSelection();

        videoFileName.textContent = file.name;
        videoDropzone.querySelector('.dropzone-content').classList.add('hidden');
        videoFileInfo.classList.remove('hidden');
    }

    videoClearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetVideoUpload();
    });

    function resetVideoUpload() {
        state.videoFile = null;
        videoFileInput.value = '';
        videoFileInfo.classList.add('hidden');
        videoDropzone.querySelector('.dropzone-content').classList.remove('hidden');
    }

    function selectVideoSample(sampleId, chipElement) {
        resetVideoUpload();
        state.selectedVideoSampleId = sampleId;

        document.querySelectorAll('#video-sample-chips .chip').forEach(c => c.classList.remove('selected'));
        chipElement.classList.add('selected');
    }

    function clearVideoSampleSelection() {
        document.querySelectorAll('#video-sample-chips .chip').forEach(c => c.classList.remove('selected'));
    }

    // -------------------------------------------------------------------------
    // Run Video Stream Analysis
    // -------------------------------------------------------------------------
    btnRunVideo.addEventListener('click', async () => {
        if (!state.videoFile && !state.selectedVideoSampleId) {
            alert('Please upload a video file or select the sample video first.');
            return;
        }

        const formData = new FormData();
        if (state.videoFile) {
            formData.append('file', state.videoFile);
        } else if (state.selectedVideoSampleId) {
            formData.append('sample_id', state.selectedVideoSampleId);
        }

        formData.append('confidence', videoConfSlider.value);
        formData.append('iou', videoIouSlider.value);

        // UI Loading State
        btnRunVideo.disabled = true;
        videoSpinner.classList.remove('hidden');
        btnRunVideo.querySelector('.btn-text').textContent = 'Processing Video Frames (CPU)...';

        try {
            const res = await fetch('/api/v1/predict/video', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || 'Video processing failed.');
            }

            renderVideoResults(data);

        } catch (err) {
            console.error('Video prediction error:', err);
            videoAlertBanner.className = 'alert-banner error';
            videoAlertBanner.textContent = `Error: ${err.message}`;
            videoAlertBanner.style.display = 'block';
            videoPlaceholder.classList.add('hidden');
            videoOutputContainer.classList.remove('hidden');
        } finally {
            btnRunVideo.disabled = false;
            videoSpinner.classList.add('hidden');
            btnRunVideo.querySelector('.btn-text').textContent = '⚡ Process Video Feed';
        }
    });

    function renderVideoResults(data) {
        videoPlaceholder.classList.add('hidden');
        videoOutputContainer.classList.remove('hidden');

        // Update Stats
        statVideoFrames.textContent = data.frames_processed;
        statVideoDetections.textContent = data.total_detections;
        statVideoFps.textContent = `${data.processing_fps} FPS`;
        statVideoLatency.textContent = `${data.average_inference_time_ms} ms`;

        // Alert Banner
        videoAlertBanner.className = 'alert-banner success';
        videoAlertBanner.textContent = `🎬 Completed video processing: ${data.frames_processed} frames in ${data.total_processing_time_s}s with ${data.total_detections} cumulative detections.`;

        // Load Video
        if (data.output_video_url) {
            annotatedVideoPlayer.src = data.output_video_url;
            annotatedVideoPlayer.load();
            annotatedVideoPlayer.play().catch(e => console.log('Autoplay prevented by browser:', e));
        }
    }

    // Initialize System
    initSystem();
});

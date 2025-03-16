// Global variables
let originalImage = null
let processedImage = null
let currentFilter = "none"
let currentEffect = null
const imageHistory = []
let isProcessing = false
const cropMode = false
let textPosition = "center"
let flipHorizontal = false
let flipVertical = false
let rotation = 0

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
  // Navigation
  const navItems = document.querySelectorAll(".nav-item")
  const sections = document.querySelectorAll(".section")
  const sectionTitle = document.getElementById("sectionTitle")

  // Image upload
  const dropArea = document.getElementById("dropArea")
  const imageUpload = document.getElementById("imageUpload")
  const uploadSection = document.getElementById("uploadSection")
  const imagePreviewContainer = document.getElementById("imagePreviewContainer")
  const originalImageEl = document.getElementById("originalImage")
  const processedImageEl = document.getElementById("processedImage")
  const processingOverlay = document.getElementById("processingOverlay")

  // View controls
  const originalViewBtn = document.getElementById("originalViewBtn")
  const splitViewBtn = document.getElementById("splitViewBtn")
  const processedViewBtn = document.getElementById("processedViewBtn")
  const comparisonSlider = document.getElementById("comparisonSlider")

  // Action buttons
  const resetButton = document.getElementById("resetButton")
  const downloadButton = document.getElementById("downloadButton")

  // Filter items
  const filterItems = document.querySelectorAll(".filter-item")

  // Adjustment sliders
  const brightnessSlider = document.getElementById("brightnessSlider")
  const contrastSlider = document.getElementById("contrastSlider")
  const saturationSlider = document.getElementById("saturationSlider")
  const sharpnessSlider = document.getElementById("sharpnessSlider")
  const blurSlider = document.getElementById("blurSlider")
  const noiseSlider = document.getElementById("noiseSlider")

  // Transform controls
  const rotationSlider = document.getElementById("rotationSlider")
  const transformButtons = document.querySelectorAll("[data-transform]")
  const resizeWidth = document.getElementById("resizeWidth")
  const resizeHeight = document.getElementById("resizeHeight")
  const resizeButton = document.getElementById("resizeButton")
  const cropButton = document.getElementById("cropButton")

  // Effect items
  const effectItems = document.querySelectorAll(".effect-item")

  // Text controls
  const textInput = document.getElementById("textInput")
  const fontSelect = document.getElementById("fontSelect")
  const fontSizeInput = document.getElementById("fontSizeInput")
  const textColorInput = document.getElementById("textColorInput")
  const positionButtons = document.querySelectorAll(".position-btn")
  const addTextButton = document.getElementById("addTextButton")

  // History
  const historyList = document.getElementById("historyList")

  // Crop modal
  const cropModal = document.getElementById("cropModal")
  const cropImage = document.getElementById("cropImage")
  const cropBox = document.getElementById("cropBox")
  const cancelCropButton = document.getElementById("cancelCropButton")
  const applyCropButton = document.getElementById("applyCropButton")
  const closeModalButton = document.querySelector(".close-modal")

  // Navigation functionality
  navItems.forEach((item) => {
    item.addEventListener("click", function () {
      const section = this.getAttribute("data-section")

      // Update active nav item
      navItems.forEach((nav) => nav.classList.remove("active"))
      this.classList.add("active")

      // Update visible section
      sections.forEach((sec) => sec.classList.remove("active"))
      document.getElementById(`${section}Section`).classList.add("active")

      // Update section title
      sectionTitle.textContent = this.querySelector("span").textContent
    })
  })

  // Drag and drop functionality
  dropArea.addEventListener("dragover", function (e) {
    e.preventDefault()
    this.classList.add("drag-over")
  })

  dropArea.addEventListener("dragleave", function () {
    this.classList.remove("drag-over")
  })

  dropArea.addEventListener("drop", function (e) {
    e.preventDefault()
    this.classList.remove("drag-over")

    if (e.dataTransfer.files.length) {
      handleImageUpload(e.dataTransfer.files[0])
    }
  })

  imageUpload.addEventListener("change", function () {
    if (this.files.length) {
      handleImageUpload(this.files[0])
    }
  })

  // Handle image upload
  function handleImageUpload(file) {
    if (!file.type.match("image.*")) {
      showToast("Error", "Please select a valid image file", "error")
      return
    }

    const reader = new FileReader()

    reader.onload = (e) => {
      originalImage = e.target.result
      processedImage = originalImage

      // Update UI
      uploadSection.classList.add("hidden")
      imagePreviewContainer.classList.remove("hidden")
      originalImageEl.src = originalImage
      processedImageEl.src = processedImage

      // Reset all controls
      resetControls()

      // Add to history
      addToHistory(originalImage, "Original")

      // Switch to filters section
      document.querySelector('[data-section="filters"]').click()

      showToast("Success", "Image uploaded successfully", "success")
    }

    reader.readAsDataURL(file)
  }

  // View controls functionality
  originalViewBtn.addEventListener("click", function () {
    this.classList.add("active")
    splitViewBtn.classList.remove("active")
    processedViewBtn.classList.remove("active")

    document.querySelector(".original-image").style.width = "100%"
    document.querySelector(".processed-image").style.width = "0%"
    comparisonSlider.style.display = "none"
  })

  splitViewBtn.addEventListener("click", function () {
    this.classList.add("active")
    originalViewBtn.classList.remove("active")
    processedViewBtn.classList.remove("active")

    document.querySelector(".original-image").style.width = "50%"
    document.querySelector(".processed-image").style.width = "50%"
    comparisonSlider.style.display = "block"
  })

  processedViewBtn.addEventListener("click", function () {
    this.classList.add("active")
    originalViewBtn.classList.remove("active")
    splitViewBtn.classList.remove("active")

    document.querySelector(".original-image").style.width = "0%"
    document.querySelector(".processed-image").style.width = "100%"
    comparisonSlider.style.display = "none"
  })

  // Comparison slider functionality
  let isDragging = false

  comparisonSlider.addEventListener("mousedown", (e) => {
    e.preventDefault()
    isDragging = true
  })

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return

    const container = document.querySelector(".image-comparison")
    const containerRect = container.getBoundingClientRect()
    const x = e.clientX - containerRect.left
    const percent = (x / containerRect.width) * 100

    if (percent >= 0 && percent <= 100) {
      document.querySelector(".original-image").style.width = `${percent}%`
      document.querySelector(".processed-image").style.width = `${100 - percent}%`
      comparisonSlider.style.left = `${percent}%`
    }
  })

  document.addEventListener("mouseup", () => {
    isDragging = false
  })

  // Reset and download buttons
  resetButton.addEventListener("click", () => {
    if (!originalImage) return

    processedImage = originalImage
    processedImageEl.src = processedImage
    resetControls()

    showToast("Info", "Image has been reset", "info")
  })

  downloadButton.addEventListener("click", () => {
    if (!processedImage) return

    const link = document.createElement("a")
    link.href = processedImage
    link.download = "processed_image_" + new Date().getTime() + ".png"
    link.click()

    showToast("Success", "Image downloaded successfully", "success")
  })

  // Filter functionality
  filterItems.forEach((item) => {
    item.addEventListener("click", function () {
      if (isProcessing || !originalImage) return

      const filter = this.getAttribute("data-filter")

      // Update UI
      filterItems.forEach((f) => f.classList.remove("active"))
      this.classList.add("active")

      // Apply filter
      currentFilter = filter
      applyImageProcessing()
    })
  })

  // Adjustment sliders functionality
  const sliders = [brightnessSlider, contrastSlider, saturationSlider, sharpnessSlider, blurSlider, noiseSlider]

  sliders.forEach((slider) => {
    slider.addEventListener("input", function () {
      // Update value display
      this.nextElementSibling.textContent = this.value
    })

    slider.addEventListener("change", () => {
      if (!originalImage) return
      applyImageProcessing()
    })
  })

  // Transform controls functionality
  rotationSlider.addEventListener("input", function () {
    this.nextElementSibling.textContent = this.value + "°"
  })

  rotationSlider.addEventListener("change", function () {
    if (!originalImage) return
    rotation = Number.parseInt(this.value)
    applyImageProcessing()
  })

  transformButtons.forEach((button) => {
    button.addEventListener("click", function () {
      if (!originalImage) return

      const transform = this.getAttribute("data-transform")

      if (transform === "flip_horizontal") {
        flipHorizontal = !flipHorizontal
        this.classList.toggle("active")
      } else if (transform === "flip_vertical") {
        flipVertical = !flipVertical
        this.classList.toggle("active")
      }

      applyImageProcessing()
    })
  })

  resizeButton.addEventListener("click", () => {
    if (!originalImage) return

    const width = Number.parseInt(resizeWidth.value)
    const height = Number.parseInt(resizeHeight.value)

    if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
      showToast("Error", "Please enter valid dimensions", "error")
      return
    }

    // Apply resize
    applyImageProcessing({
      resize: true,
      width: width,
      height: height,
    })
  })

  cropButton.addEventListener("click", () => {
    if (!originalImage) return

    // Show crop modal
    cropModal.classList.add("active")
    cropImage.src = processedImage

    // Initialize crop box
    setTimeout(() => {
      const imgWidth = cropImage.offsetWidth
      const imgHeight = cropImage.offsetHeight

      cropBox.style.width = `${imgWidth * 0.8}px`
      cropBox.style.height = `${imgHeight * 0.8}px`
      cropBox.style.left = `${imgWidth * 0.1}px`
      cropBox.style.top = `${imgHeight * 0.1}px`

      // Add crop handles
      const handles = ["tl", "tr", "bl", "br"]
      cropBox.innerHTML = ""

      handles.forEach((pos) => {
        const handle = document.createElement("div")
        handle.className = `crop-handle ${pos}`
        cropBox.appendChild(handle)
      })

      initCropDrag()
    }, 100)
  })

  // Crop modal functionality
  function initCropDrag() {
    const cropBox = document.getElementById("cropBox")
    const cropImage = document.getElementById("cropImage")

    let isDragging = false
    let isResizing = false
    let currentHandle = null
    let startX, startY, startWidth, startHeight, startLeft, startTop

    // Drag functionality
    cropBox.addEventListener("mousedown", (e) => {
      if (e.target.classList.contains("crop-handle")) {
        // Resize
        isResizing = true
        currentHandle = e.target.classList[1] // tl, tr, bl, br
      } else {
        // Move
        isDragging = true
      }

      startX = e.clientX
      startY = e.clientY
      startWidth = Number.parseInt(cropBox.style.width)
      startHeight = Number.parseInt(cropBox.style.height)
      startLeft = Number.parseInt(cropBox.style.left)
      startTop = Number.parseInt(cropBox.style.top)

      e.preventDefault()
    })

    document.addEventListener("mousemove", (e) => {
      if (!isDragging && !isResizing) return

      const dx = e.clientX - startX
      const dy = e.clientY - startY

      if (isDragging) {
        // Move crop box
        let newLeft = startLeft + dx
        let newTop = startTop + dy

        // Constrain to image boundaries
        const maxLeft = cropImage.offsetWidth - cropBox.offsetWidth
        const maxTop = cropImage.offsetHeight - cropBox.offsetHeight

        newLeft = Math.max(0, Math.min(newLeft, maxLeft))
        newTop = Math.max(0, Math.min(newTop, maxTop))

        cropBox.style.left = `${newLeft}px`
        cropBox.style.top = `${newTop}px`
      } else if (isResizing) {
        // Resize crop box
        let newWidth = startWidth
        let newHeight = startHeight
        let newLeft = startLeft
        let newTop = startTop

        if (currentHandle === "tl") {
          newWidth = startWidth - dx
          newHeight = startHeight - dy
          newLeft = startLeft + dx
          newTop = startTop + dy
        } else if (currentHandle === "tr") {
          newWidth = startWidth + dx
          newHeight = startHeight - dy
          newTop = startTop + dy
        } else if (currentHandle === "bl") {
          newWidth = startWidth - dx
          newHeight = startHeight + dy
          newLeft = startLeft + dx
        } else if (currentHandle === "br") {
          newWidth = startWidth + dx
          newHeight = startHeight + dy
        }

        // Enforce minimum size
        newWidth = Math.max(50, newWidth)
        newHeight = Math.max(50, newHeight)

        // Constrain to image boundaries
        newLeft = Math.max(0, newLeft)
        newTop = Math.max(0, newTop)

        if (newLeft + newWidth > cropImage.offsetWidth) {
          newWidth = cropImage.offsetWidth - newLeft
        }

        if (newTop + newHeight > cropImage.offsetHeight) {
          newHeight = cropImage.offsetHeight - newTop
        }

        cropBox.style.width = `${newWidth}px`
        cropBox.style.height = `${newHeight}px`
        cropBox.style.left = `${newLeft}px`
        cropBox.style.top = `${newTop}px`
      }
    })

    document.addEventListener("mouseup", () => {
      isDragging = false
      isResizing = false
      currentHandle = null
    })
  }

  cancelCropButton.addEventListener("click", () => {
    cropModal.classList.remove("active")
  })

  closeModalButton.addEventListener("click", () => {
    cropModal.classList.remove("active")
  })

  applyCropButton.addEventListener("click", () => {
    // Get crop dimensions
    const cropBox = document.getElementById("cropBox")
    const cropImage = document.getElementById("cropImage")

    const imgRect = cropImage.getBoundingClientRect()
    const boxRect = cropBox.getBoundingClientRect()

    const scaleX = cropImage.naturalWidth / cropImage.width
    const scaleY = cropImage.naturalHeight / cropImage.height

    const cropData = {
      x: Number.parseInt(cropBox.style.left) * scaleX,
      y: Number.parseInt(cropBox.style.top) * scaleY,
      width: Number.parseInt(cropBox.style.width) * scaleX,
      height: Number.parseInt(cropBox.style.height) * scaleY,
    }

    // Apply crop
    applyImageProcessing({
      crop: true,
      cropData: cropData,
    })

    // Close modal
    cropModal.classList.remove("active")
  })

  // Effect functionality
  effectItems.forEach((item) => {
    item.addEventListener("click", function () {
      if (isProcessing || !originalImage) return

      const effect = this.getAttribute("data-effect")

      // Update UI
      effectItems.forEach((e) => e.classList.remove("active"))
      this.classList.add("active")

      // Apply effect
      currentEffect = effect
      applyImageProcessing()
    })
  })

  // Text functionality
  positionButtons.forEach((button) => {
    button.addEventListener("click", function () {
      positionButtons.forEach((b) => b.classList.remove("active"))
      this.classList.add("active")

      textPosition = this.getAttribute("data-position")
    })
  })

  addTextButton.addEventListener("click", () => {
    if (!originalImage || !textInput.value.trim()) return

    const text = textInput.value.trim()
    const font = fontSelect.value
    const fontSize = fontSizeInput.value
    const color = textColorInput.value

    // Apply text
    applyImageProcessing({
      addText: true,
      text: text,
      font: font,
      fontSize: fontSize,
      color: color,
      position: textPosition,
    })

    // Clear text input
    textInput.value = ""
  })

  // Apply image processing
  function applyImageProcessing(options = {}) {
    if (!originalImage || isProcessing) return

    isProcessing = true
    processingOverlay.classList.remove("hidden")

    // Get all adjustment values
    const adjustments = {
      brightness: Number.parseInt(brightnessSlider.value),
      contrast: Number.parseInt(contrastSlider.value),
      saturation: Number.parseInt(saturationSlider.value),
      sharpness: Number.parseInt(sharpnessSlider.value),
      blur: Number.parseInt(blurSlider.value),
      noise: Number.parseInt(noiseSlider.value),
      rotation: rotation,
      flipHorizontal: flipHorizontal,
      flipVertical: flipVertical,
      filter: currentFilter,
      effect: currentEffect,
      ...options,
    }

    // Send to server for processing
    fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: processedImage,
        adjustments: adjustments,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          showToast("Error", data.error, "error")
          return
        }

        // Update processed image
        processedImage = data.processed_image
        processedImageEl.src = processedImage

        // Add to history
        let actionName = "Processed"

        if (adjustments.filter && adjustments.filter !== "none") {
          actionName = `Filter: ${adjustments.filter}`
        } else if (adjustments.effect) {
          actionName = `Effect: ${adjustments.effect}`
        } else if (adjustments.addText) {
          actionName = "Added Text"
        } else if (adjustments.crop) {
          actionName = "Cropped"
        } else if (adjustments.resize) {
          actionName = "Resized"
        }

        addToHistory(processedImage, actionName)

        showToast("Success", "Image processed successfully", "success")
      })
      .catch((error) => {
        console.error("Error:", error)
        showToast("Error", "Failed to process image", "error")
      })
      .finally(() => {
        isProcessing = false
        processingOverlay.classList.add("hidden")
      })
  }

  // Add to history
  function addToHistory(imageData, actionName) {
    imageHistory.push({
      image: imageData,
      action: actionName,
      timestamp: new Date().toLocaleTimeString(),
    })

    updateHistoryUI()
  }

  // Update history UI
  function updateHistoryUI() {
    historyList.innerHTML = ""

    if (imageHistory.length === 0) {
      historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-history"></i>
                    <p>Your editing history will appear here</p>
                </div>
            `
      return
    }

    imageHistory.forEach((item, index) => {
      const historyItem = document.createElement("div")
      historyItem.className = "history-item"
      historyItem.innerHTML = `
                <img src="${item.image}" class="history-image" alt="History ${index}">
                <div class="history-info">
                    <div>${item.action}</div>
                    <div>${item.timestamp}</div>
                </div>
            `

      historyItem.addEventListener("click", () => {
        processedImage = item.image
        processedImageEl.src = processedImage

        showToast("Info", `Reverted to: ${item.action}`, "info")
      })

      historyList.appendChild(historyItem)
    })
  }

  // Reset controls
  function resetControls() {
    // Reset filter selection
    filterItems.forEach((item) => item.classList.remove("active"))
    document.querySelector('[data-filter="none"]').classList.add("active")
    currentFilter = "none"

    // Reset effect selection
    effectItems.forEach((item) => item.classList.remove("active"))
    currentEffect = null

    // Reset sliders
    brightnessSlider.value = 0
    brightnessSlider.nextElementSibling.textContent = "0"

    contrastSlider.value = 0
    contrastSlider.nextElementSibling.textContent = "0"

    saturationSlider.value = 0
    saturationSlider.nextElementSibling.textContent = "0"

    sharpnessSlider.value = 0
    sharpnessSlider.nextElementSibling.textContent = "0"

    blurSlider.value = 0
    blurSlider.nextElementSibling.textContent = "0"

    noiseSlider.value = 0
    noiseSlider.nextElementSibling.textContent = "0"

    rotationSlider.value = 0
    rotationSlider.nextElementSibling.textContent = "0°"
    rotation = 0

    // Reset transforms
    flipHorizontal = false
    flipVertical = false
    document.querySelectorAll("[data-transform]").forEach((btn) => btn.classList.remove("active"))

    // Reset resize inputs
    resizeWidth.value = ""
    resizeHeight.value = ""

    // Reset text inputs
    textInput.value = ""
    fontSizeInput.value = "24"
    textColorInput.value = "#ffffff"
    positionButtons.forEach((btn) => btn.classList.remove("active"))
    document.querySelector('[data-position="center"]').classList.add("active")
    textPosition = "center"
  }

  // Toast notifications
  function showToast(title, message, type = "info") {
    const toastContainer = document.getElementById("toastContainer")

    const toast = document.createElement("div")
    toast.className = `toast ${type}`

    let icon = "info-circle"
    if (type === "success") icon = "check-circle"
    if (type === "error") icon = "exclamation-circle"
    if (type === "warning") icon = "exclamation-triangle"

    toast.innerHTML = `
            <div class="toast-icon">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">&times;</button>
        `

    toastContainer.appendChild(toast)

    // Auto remove after 5 seconds
    setTimeout(() => {
      toast.style.animation = "slideOut 0.3s ease forwards"
      setTimeout(() => {
        toast.remove()
      }, 300)
    }, 5000)

    // Close button
    toast.querySelector(".toast-close").addEventListener("click", () => {
      toast.style.animation = "slideOut 0.3s ease forwards"
      setTimeout(() => {
        toast.remove()
      }, 300)
    })
  }
})


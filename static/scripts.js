document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadedImage = document.getElementById('uploaded-image');
    const applyOutfitButton = document.getElementById('apply-outfit');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.image) {
            uploadedImage.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" alt="Uploaded Image">`;
        } else {
            alert('Error uploading image');
        }
    });

    applyOutfitButton.addEventListener('click', async () => {
        const outfitType = document.getElementById('outfit-type').value;
        const outfitSize = document.getElementById('outfit-size').value;
        const outfitColor = document.getElementById('outfit-color').value;

        const response = await fetch('/change_outfit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: outfitType,
                size: outfitSize,
                color: outfitColor
            })
        });

        const data = await response.json();
        if (!data.success) {
            alert('Error applying outfit');
        }
    });
});
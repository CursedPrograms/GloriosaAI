interface Artwork {
    id: number;
    title: string;
    imageUrl: string;
  }
  
  const artworks: Artwork[] = [
    { id: 1, title: 'Abstract Art 1', imageUrl: 'path/to/image1.jpg' },
    { id: 2, title: 'Digital Landscape', imageUrl: 'path/to/image2.jpg' },
    // Add more artworks
  ];
  
  function displayArtworks() {
    const galleryElement = document.getElementById('artGallery');
  
    if (!galleryElement) {
      console.error('Gallery element not found.');
      return;
    }
  
    artworks.forEach((artwork) => {
      const card = document.createElement('div');
      card.classList.add('artwork-card');
  
      const image = document.createElement('img');
      image.src = artwork.imageUrl;
      image.alt = artwork.title;
  
      const title = document.createElement('p');
      title.textContent = artwork.title;
  
      card.appendChild(image);
      card.appendChild(title);
  
      galleryElement.appendChild(card);
    });
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    displayArtworks();
  });
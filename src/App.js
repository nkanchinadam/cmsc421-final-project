import React, { useState } from 'react';
import './App.css';

const Modal = ({ isOpen, onClose, results }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        {results}
        <button onClick={onClose}>Done</button>
      </div>
    </div>
  );
};

function App() {
  const [selectedImg, setSelectedImg] = useState(null);
  const [isModalOpen, setModalOpen] = useState(false);

  const openModal = () => setModalOpen(true);
  const closeModal = () => setModalOpen(false);

  function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
      setSelectedImg(reader.result);
    };
    reader.readAsDataURL(event.target.files[0]);
  }

  function handleUpload() {
    openModal();
  }

  return (
    <div className="bodyDiv">
      <h1>Upload A Movie Poster to Get The Genre!</h1>
      <div id="imageContainer">
        {selectedImg && 
          <img id="selectedImg" src={selectedImg} alt="Selected Image" />
        }
      </div>
      {isModalOpen && <Modal isOpen={isModalOpen} onClose={closeModal}>
        <p>The genre is</p>
      </Modal>}

      <form encType="multipart/form-data">
        <input type="file" name="file" id="chooseFileBtn" onChange={previewImage} />
        <input id="uploadBtn" type="button" value="upload" disabled={selectedImg==null} onClick={handleUpload} />
      </form>
    </div>
  );
}

export default App;

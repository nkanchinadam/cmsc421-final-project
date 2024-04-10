import React, { useState } from 'react';
import './App.css';

function App() {

    const [selectedImg, setSelectedImg] = useState(null);

    function previewImage(event) {
        const reader = new FileReader();
        reader.onload = function () {
            setSelectedImg(reader.result);
        };
        reader.readAsDataURL(event.target.files[0]);
    }

  return (
    <div className="bodyDiv">
    <h1>Upload A Movie Poster to Get The Genre!</h1>
    <div id="imageContainer">
      {selectedImg && 
        <img id="selectedImg" src={selectedImg} alt="Selected Image" />
      }
    </div>
    <form action="/predict" method="post" encType="multipart/form-data">
        <input type="file" name="file" id="chooseFileBtn" onChange={previewImage} />
        <input id="uploadBtn" type="submit" value="upload" disabled={selectedImg==null} />
    </form>
</div>
  );

}



export default App;

import React, { useState } from 'react';
import axios from '../services/api';

function UploadForm() {
    const [file, setFile] = useState(null);

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await axios.post('/upload', formData);
            console.log(response.data); // Выводим отчет
        } catch (error) {
            console.error('Ошибка загрузки:', error);
        }
    };

    return (
        <div>
            <input type="file" onChange={(e) => setFile(e.target.files[0])} />
            <button onClick={handleUpload}>Загрузить</button>
        </div>
    );
}

export default UploadForm;

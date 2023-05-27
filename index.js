const express = require('express');
const app = express();
const port = 3000;
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const fileUpload = require('express-fileupload');

app.use('/model', express.static(path.join(__dirname, 'model')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload());

const loadModel = async () => {
    const model = await tf.loadLayersModel('http://localhost:3000/model/model.json');
    return model;
}

app.post('/predict', async (req, res) => {
    console.log(req.files);
    if (!req.files) {
        return res.status(400).send('No files were uploaded.');
    } else if (req.files.image.mimetype !== 'image/jpeg' && req.files.image.mimetype !== 'image/png') {
        return res.status(400).send('Invalid file type.');
    }
    else {
        const model = await loadModel();
        const img = tf.node.decodeImage(req.files.image.data, 3);
        const resized = tf.image.resizeBilinear(img, [150, 150]);
        const expanded = resized.expandDims(0);
        const prediction = await model.predict(expanded).data();
        const result = prediction[0] > 0.5 ? 'Unhealthy' : 'Healthy';
        
        tf.dispose(img);
        tf.dispose(resized);
        tf.dispose(expanded);
        
        res.json({ result });

    }
});

app.listen(port, () => console.log(`Server listening on port ${port}!`));
```markdown
# Resume Screening Application

Welcome to the **Resume Screening Application**! This is an end-to-end machine learning pipeline designed to automatically classify resumes based on job categories. This project leverages advanced NLP models to analyze resumes and predict the appropriate job category, which can be helpful for recruiters and job seekers alike.

## 🚀 Demo Video

Check out the demo video of the project on YouTube:

[![Resume Screening Application Demo](https://img.youtube.com/vi/V8uE1iyyn3A/0.jpg)](https://youtu.be/V8uE1iyyn3A)

Click on the image above or [this link](https://youtu.be/V8uE1iyyn3A) to watch the full demo.

## 🌟 Features
- Upload your resume (PDF format).
- Automatic classification of the resume into a predicted job category.
- User-friendly web interface built with Flask.
- Robust ML pipeline for data ingestion, preprocessing, and model training.
- Pre-trained model and utility files for fast execution.

## 💻 Installation Instructions

To get the project up and running on your local machine, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/arhamkhan779/ResumeScreeningApplication.git
```

### 2. Install dependencies:

Navigate to the project folder and install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the Flask app:

Now, you can start the application by running:

```bash
python app.py
```

The app will be running locally, and you can access it in your browser by visiting:

```
https://localhost:5000
```

## 🛠️ Project Workflows

### Workflows Overview
The following workflows are part of the project's pipeline:

1. **Update config.yaml**: Modify the `config.yaml` file to set up any application-specific configurations such as paths, parameters, etc.
2. **Update secrets.yaml**: (Optional) Add any secret configurations such as API keys, access credentials, etc.
3. **Update params.yaml**: Update any training parameters like learning rate, batch size, etc., depending on your model training.
4. **Update the entity**: This involves defining the necessary components for the machine learning pipeline such as `DataIngestion`, `DataPreprocessing`, and model entities.
5. **Update the configuration manager**: Update the configuration manager within `src/config` to manage application settings.
6. **Update components**: Modify the `components` directory files to accommodate any custom changes in the pipeline.
7. **Update the pipeline**: Ensure all the pipeline files in `src/pipeline` are updated for the smooth execution of tasks.
8. **Update the `main.py`**: The `main.py` file is the entry point for executing the application. Ensure it includes all necessary imports and function calls.
9. **Update `dvc.yaml`**: If using DVC (Data Version Control), update the `dvc.yaml` to handle datasets, models, and pipelines.

### File Structure:
Here’s an overview of the project directory structure:

```plaintext
├── .github
├── artifacts
│   ├── Model
│   ├── model.h5
│   ├── data_ingestion
│   └── ...
├── components
│   ├── DARA_PREPROCESSING.py
│   ├── DATA_INGESTION.py
│   ├── MODEL.py
│   ├── MODEL_TRAINING.py
│   └── ...
├── config
│   ├── config.yaml
│   ├── secrets.yaml
│   └── ...
├── pipeline
│   ├── DATA_INGESTION_PIPELINE.py
│   ├── DATA_PREPROCESSING_PIPELINE.py
│   ├── MODEL_TRAINER_PIPELINE.py
│   └── ...
├── templates
│   ├── index.html
│   └── pp.jpg
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 📝 Detailed Project Pipeline

The project follows a structured pipeline with distinct components:

1. **Data Ingestion**: 
   - Extract raw resume data from uploaded files.
   - Store data in a usable format for preprocessing.
   
2. **Data Preprocessing**: 
   - Clean the resume data (remove stop words, normalize text, etc.).
   - Feature extraction (e.g., using TF-IDF or word embeddings).
   
3. **Model Training**: 
   - Train a machine learning model (e.g., a neural network, or any other classification model).
   - Evaluate the model using metrics such as accuracy, precision, and recall.

4. **Model Deployment**:
   - The trained model is deployed through the Flask app for real-time resume classification.

### Key Files:
- **`app.py`**: The main Flask application that handles the front-end and back-end.
- **`main.py`**: Contains the entry point for model training and running the application.
- **`config.yaml`**: Contains all configuration settings related to the model and data.
- **`dvc.yaml`**: If you're using DVC, this file tracks the data, models, and pipelines.

## 📊 Evaluation Metrics:
The model's performance is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**

Graphs for these metrics are saved as PNG files (`model_Accuracy.png`, `model_Precision.png`, `model_Recall.png`, etc.) in the `artifacts` folder.

## 🔑 Configuration

The following configuration files can be updated to suit your needs:

- **`config.yaml`**: Update paths and other settings for the application.
- **`secrets.yaml`**: (Optional) Store any sensitive information such as API keys or credentials.
- **`params.yaml`**: Customize model training hyperparameters like learning rate, batch size, etc.
  
## 👥 Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you have any issues or questions, feel free to open an issue in this repository.

Thank you for checking out the **Resume Screening Application**! We hope this helps automate resume screening and makes the hiring process faster and more efficient!
```

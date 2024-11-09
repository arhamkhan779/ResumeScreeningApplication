from RESUMESCREENINGAPP.entity.config_entity import TrainingConfig
from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP import logger
from tensorflow import keras
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.metrics import confusion_matrix

class ModelTrainer:
    def __init__(self,config: TrainingConfig):
        self.config=config
        
        try:
            logger.info("Loading Text Processor Pipeline ---- > start")
            self.text_processor=joblib.load(self.config.text_preprocessor_path)
            logger.info("Loading Text Processor Pipeline ---- > Done")

            logger.info("Loading Target Processor Pipeline ---- > start")
            self.target_processor=joblib.load(self.config.target_preprocessor_path)
            logger.info("Loading Target Processor Pipeline ---- > Done")

            logger.info("Loading Base Model ---- > start")
            self.model=joblib.load(self.config.base_model)
            logger.info("Loading Base Model ---- > Done")

            logger.info("Loading Dataset ---- > start")
            self.Data_Frame=pd.read_csv(self.config.data_set_dir)
            logger.info("Loading Dataset ---- > completed")

        except Exception as e:
            logger.info(e)
            raise e
        

    def Train_Model_On_Custom_Dataset(self):
        try:
            logger.info(f"Model Training Process on Dataset with Shape {self.Data_Frame.shape} and Columns {self.Data_Frame.columns}")

            logger.info("Transforming Text Column -----> Start")
            X=self.text_processor.fit_transform(self.Data_Frame['Resume'])
            logger.info("Transforming Text Column -----> Completed")
            logger.info(f"Shape : {X.shape} Column Data After Transformation {X[0]}")

            logger.info("Transforming Target Column -----> Start")
            Y=self.target_processor.fit_transform(self.Data_Frame['Category'])
            logger.info(f"Transforming Target Column -----> Completed")
            logger.info(f"Shape : {Y.shape} Column Data After Transformation {Y[0]}")
            
            logger.info("Splitting Data into Training and Testing Part")
            self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(X,Y,test_size=0.20,random_state=45)
            logger.info(f"X_train : {self.X_train.shape} X_test: {self.X_test.shape} Y_train: {self.Y_train.shape} Y_test: {self.Y_test.shape}")
            self.model.compile(optimizer='Adam',loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy','precision','recall'])
            logger.info(f"Start training on Epochs : {self.config.epochs} , Batch_Size : {self.config.batch}")
            history=self.model.fit(self.X_train,self.Y_train,batch_size=self.config.batch,epochs=self.config.epochs,validation_data=(self.X_test,self.Y_test))

            logger.info(f"Saving History as results.csv at {self.config.results_path}")
            df=pd.DataFrame(history.history)
            df.to_csv(self.config.results_path)
            logger.info(f"Saving trained model at {self.config.trained_model_path}")
            self.model.save(self.config.trained_model_path)

        except Exception as e:
            logger.info(e)
            raise e
        
    def plot_results(self,X,Y,Y_val,label,label_val,y_label,x_label,title,Path):
            plt.figure(figsize=(10, 6))
            plt.plot(X, Y, label=label)
            plt.plot(X, Y_val, label=label_val)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.xlim(left=0)  # This sets the x-axis to start at 0 (epochs)
            plt.ylim(bottom=0)
            plt.legend()
        # Save the loss plot
            loss_plot_path = Path
            plt.savefig(loss_plot_path)
            plt.close()  

    
    def Save_Plot_Results(self):
        try:
            logger.info("Plotting Results")
            results=pd.read_csv(self.config.results_path)
            epochs=[i for i in range(1,self.config.epochs+1)]
            accuracy=[(round(i,1))*100 for i in results['accuracy']]
            val_accuracy=[(round(i,1))*100 for i in results['val_accuracy']]
            loss=[i for i in results['loss']]
            val_loss=[i for i in results['val_loss']]

            precision=[i for i in results['precision']]
            val_precision=[i for i in results['val_precision']]

            recall=[i for i in results['recall']]
            val_recall=[i for i in results['val_recall']]
            
            logger.info("Plotting Accuracy")
            self.plot_results(epochs,accuracy,val_accuracy,"Accuracy","Validation Accuracy","Accuracy","Epochs","Model Accuracy","artifacts/training/model_Accuracy.png")
            
            logger.info("Plotting Loss")
            self.plot_results(epochs,loss,val_loss,"Loss","Validation Loss","Loss","Epochs","Model Loss","artifacts/training/model_loss.png")
            
            logger.info("Plotting Precision")
            self.plot_results(epochs,precision,val_precision,"Precision","Validation Precision","Precision","Epochs","Model Precision","artifacts/training/model_Precision.png")
            
            logger.info("Plotting Recall ")
            self.plot_results(epochs,recall,val_recall,"Recall","Validation Recall","Recall","Epochs","Model Recall","artifacts/training/model_Recall.png")

            logger.info("Plotting Confusion Matrix")
            prediction=self.model.predict(self.X_test)
            
            Y_test_labels = np.argmax(self.Y_test, axis=1)
            Y_pred_labels = np.argmax(prediction, axis=1)

            cm = confusion_matrix(Y_test_labels, Y_pred_labels)
            plt.figure(figsize=(30, 15))  
            sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.arange(0, 25),  # 24 classes, starting from 1 to 24
            yticklabels=np.arange(0, 25),
            cbar=True, linewidths=0.5)
            plt.title("Confusion Matrix", fontsize=16)
            plt.xlabel("Predicted Labels", fontsize=12)
            plt.ylabel("True Labels", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0, va="center")
            output_path = 'artifacts/training/confusion_matrix.png'
            plt.tight_layout()  # Adjust layout to avoid clipping
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            logger.info(e)
            raise e

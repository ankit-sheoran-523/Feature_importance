import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

def get_model(task_type):
    if task_type == "Classification":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42)
    
def evaluate_model(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    
    if task_type == "Classification":
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_test, y_pred,output_dict=True)
        report_df=pd.DataFrame(report).transpose().round(3)
        report_df=report_df.reset_index().rename(columns={"index":"Class"})
        
        matrix=confusion_matrix(y_test, y_pred)
        labels=unique_labels(y_test,y_pred)
        matrix_df=pd.DataFrame(matrix,index=[f"Actual {i}" for i in labels],columns=[f"Predicted {i}" for i in labels])
        
        accuracy=accuracy_score(y_test,y_pred)
        return report_df,matrix_df,accuracy
    
    else:
        
        eval_result = {
            "MSE": round(mean_squared_error(y_test, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            "R2": round(r2_score(y_test, y_pred) * 100, 3)
        }
        
        score = pd.DataFrame.from_dict(eval_result, orient='index', columns=['Score'])
        score=score.reset_index()
        score.columns=["Metric","Score"]
        return score


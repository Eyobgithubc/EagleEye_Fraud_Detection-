import shap
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.model_selection import train_test_split
def prepare_data(data, target_column='class', test_size=0.2, random_state=42):
   
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)





def explain_with_shap(model, X_train, X_test, feature_name=None, sample_size=100):
 
    
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
       
        explainer = shap.KernelExplainer(model.predict, X_train.sample(sample_size))
        shap_values = explainer.shap_values(X_test.sample(sample_size))
    

    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)


    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    plt.show()


    if feature_name and feature_name in X_test.columns:
        shap.dependence_plot(feature_name, shap_values, X_test)


def explain_with_lime(model, X_train, X_test, instance_index):
 

    lime_explainer = lime_tabular.LimeTabularExplainer(X_train.values, 
                                                        feature_names=X_train.columns, 
                                                        class_names=['Not Fraud', 'Fraud'], 
                                                        mode='classification')

    instance = X_test.iloc[instance_index]

 
    lime_exp = lime_explainer.explain_instance(instance.values, 
                                                model.predict_proba, 
                                                num_features=5)

   
    lime_exp.show_in_notebook(show_table=True)

    lime_exp.as_pyplot_figure()
    plt.title(f"LIME Feature Importance for Instance {instance_index}")
    plt.show()


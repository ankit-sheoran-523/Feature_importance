import streamlit as st
import pandas as pd
import numpy as np
import time
from functions import get_selected_dataframe,preprocess_with_column_names
from TaskDetection import detect_task_type,preprocess_target
from model import evaluate_model,get_model
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import RFE, mutual_info_classif,mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.express as px
import plotly.graph_objects as go









st.title("**Interactive Feature Selection & Model Evaluation Tool**")
st.markdown("A Streamlit Interactive tool for task-aware feature selection, visual analysis, and model evaluation")

source = st.selectbox("**Select data source**", ["None", "User Dataset", "Preload Dataset"])

uploaded_file = None
demo_choice = None

if source == "User Dataset":
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    placeholder = st.empty()
    with placeholder:
        st.info("Make sure the dataset must have **_Target column_** at last.")
    time.sleep(5)
    placeholder.empty()
    

if source == "Preload Dataset":
    demo_choice = st.selectbox("**Select Dataset**", [
        "None", "Iris (C)", "Breast Cancer (C)", "Wine (C)", "Diabetes (R)", "California Housing (R)"
    ])

data, label = get_selected_dataframe(source, uploaded_file, demo_choice)

if data is not None:
    placeholder = st.empty()
    with placeholder:
        st.success("Data has been loaded successfully")
    time.sleep(3)
    placeholder.empty()
    
    if len(data)>1000:
            data.drop_duplicates(inplace=True)
            data=data.sample(n=1000,random_state=42)
            placeholder = st.empty()
            with placeholder:
                st.info("We are using 1000 rows for speedup the operations")
            time.sleep(3)
            placeholder.empty()
    
    
    st.success(f" Loaded: {label}")
    st.write(" Preview:", data.head())
    st.empty()

    st.markdown("Choose Data Summary Type:")
    tab1,tab2=st.tabs(["Describe","Info"])
    with tab1:
        st.write(data.describe())
    with tab2:
        summary = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values, 
            "Non-Null Count": data.notnull().sum().values,
            "Missing Values": data.isnull().sum().values,
            "Unique Values": data.nunique().values
        })
        st.dataframe(summary)
        
        
    st.markdown(f"DataFrame shape : `{data.shape[0]} rows × {data.shape[1]} columns `")
    
    target_col=data.columns[-1]
    feature_cols=data.columns[:-1].tolist()
    x_raw=data[feature_cols]
    y=data[target_col]
    col_table=pd.DataFrame({"Column Type":["Features","Target/Label"],\
        "Columns":[" ,  ".join(feature_cols),target_col]})
    st.dataframe(col_table)
    
    X_processed, y, feature_names_out, col_types, preprocessor = preprocess_with_column_names(data)

    rows, cols = X_processed.shape
    st.markdown(f"Transformed shape: `{rows} rows × {cols} columns`")
    
    st.markdown(" Feature names:")
    feature_table = pd.DataFrame(feature_names_out, columns=["Feature Name"])
    st.dataframe(feature_table)

        
    st.markdown(f" Column types:")
    col_types_table = pd.DataFrame({
        "Column Type": ["Numeric", "Categorical", "Target"],
        "Columns": [
            " ,   ".join(col_types["numeric"]),
            " ,   ".join(col_types["categorical"]),
            col_types["target"]
        ]
    })
    st.dataframe(col_types_table,hide_index=True)

    y_encoded,task_type,clas_names=preprocess_target(y)
    model=get_model(task_type)
    x_train,x_test,y_train,y_test=train_test_split(X_processed,y_encoded,test_size=0.20,random_state=42)
    model.fit(x_train,y_train)
    st.success(f"{model} Model is trained")
    
    st.subheader("Default Model Evaluations are :")
    
    if task_type=="Regression":
        default_score=evaluate_model(model,x_test,y_test,task_type)
        
        st.dataframe(default_score)
    
    else :
        default_report_df, default_matrix_df,default_accuracy = evaluate_model(model,x_test,y_test,task_type)
        tab1, tab2 = st.tabs(["Classification Report", "Confusion Matrix"])
        with tab1:
            st.dataframe(default_report_df,hide_index=True)

        with tab2:
            st.dataframe(default_matrix_df)
            
        st.metric("Accuracy", f"{(default_accuracy * 100):.2f}%")



# #############################################################################################################################################################
    st.subheader("Manual Feature Selection & Evaluation")
    st.markdown("Mutual Feature Selection")
    mi_scores=(mutual_info_classif(X_processed,y_encoded) if task_type=="Classification" 
              else mutual_info_regression(X_processed,y_encoded))
    mi_df=pd.DataFrame({"Feature":feature_names_out,
                        "Mutual Info":np.round(mi_scores,4)}).sort_values(by="Mutual Info",ascending=False)
    
    st.dataframe(mi_df,use_container_width=True)
    
    st.markdown("Select Feature to train the model")
    placeholder = st.empty()
    with placeholder:
        st.warning("Select features with strong correlation to the target for better model performance. Weakly correlated features may degrade results.")
    time.sleep(5)
    placeholder.empty()
    
    selected_feature=st.multiselect(
        "Choose feature to include:",
        options=mi_df["Feature"].tolist(),
        default=mi_df["Feature"].tolist()
    )

    x_df=pd.DataFrame(X_processed,y_encoded,columns=feature_names_out)
    x_selected=x_df[selected_feature].values
    
        
        # heatmap for selected features #############################################################
    st.markdown("Correlation Heatmap of Selected Features")
    
    X_df = pd.DataFrame(X_processed, columns=feature_names_out)
    x_selected_hm = X_df[selected_feature].copy()
    include_target = st.checkbox("Include target column in heatmap?", value=False)
    if include_target:
        x_selected_hm["target"] = y_encoded 
    corr = x_selected_hm.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.xticks(rotation=45, ha='right',)

    # st.pyplot(fig)
    with st.expander(" View Feature Correlation Heatmap"):
        st.pyplot(fig)
    
    
    
    x_train_m,x_test_m,y_train_m,y_test_m=train_test_split(x_selected,y_encoded,test_size=0.2,random_state=42)
    model.fit(x_train_m,y_train_m)
    st.success(f"{model} is trained for selected columns.")
    
    st.write("Selected column Model Evaluation are :")
    
    if task_type == "Regression":
        manual_score=evaluate_model(model,x_test_m,y_test_m,task_type)
        st.dataframe(manual_score)
        
    else:
        selected_report_df,selected_matrix_df,manual_accuracy=evaluate_model(model,x_test_m,y_test_m,task_type)
        tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
        with tab1:
            st.dataframe(selected_report_df,hide_index=True)
        with tab2:
            st.dataframe(selected_matrix_df)
            
        
        st.metric("Accuracy", f"{(manual_accuracy * 100):.2f}%")



        
################## wrapper method ################################

    st.subheader("Wrapper Methods For Feature Selection")
    base_model=get_model(task_type)
    base_model.fit(X_processed,y_encoded)
    importances=base_model.feature_importances_
    
    st.markdown("Feature importance from base model")
    fig_imp,ax_imp=plt.subplots(figsize=(10,6))
    pd.Series(importances,index=feature_names_out).sort_values().plot.barh(ax=ax_imp,color="purple")
    st.pyplot(fig_imp)
    st.markdown("Select the feature selection method")
    n_features=st.slider("Select number of features",1,len(feature_names_out),value=2)
    
    tab_topN,tab_rfe,tab_forward,tab_backward=st.tabs(["Top N Features","RFE","Forward Elimination","Backward Elimination"])
    
    with tab_topN:
        st.markdown("Top N Features")
        top_indices=np.argsort(importances)[::-1][:n_features]
        selected_feature_sfm=np.array(feature_names_out)[top_indices]

        st.markdown("Selected Features are:")
        st.code(", ".join(selected_feature_sfm))
        x_df=pd.DataFrame(X_processed,columns=feature_names_out)
        x_subset_sfm=x_df[selected_feature_sfm]
        
        x_train_sfm,x_test_sfm,y_train_sfm,y_test_sfm=train_test_split(x_subset_sfm,y_encoded,test_size=0.20,random_state=42)
        model=get_model(task_type)
        model.fit(x_train_sfm,y_train_sfm)
        st.success("Model is trained for selected  columns")
        
        
        st.markdown("Correlation Heatmap of Selected Features")
    
        
        include_target_sfm = st.checkbox("Include target column in heatmap?", value=False, key="checkbox_sfm")
        if include_target_sfm:
            x_subset_sfm["target"] = y_encoded
        corr = x_subset_sfm.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View Feature Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            topN_score=evaluate_model(model,x_test_sfm,y_test_sfm,task_type)
            st.dataframe(topN_score)
        
        else:
            selected_report_df,selected_matrix_df,topN_accuracy=evaluate_model(model,x_test_sfm,y_test_sfm,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(topN_accuracy * 100):.2f}%")
        
    with tab_rfe:
        st.markdown("Recursive Feature Elimination (RFE)")
        rfe=RFE(get_model(task_type),n_features_to_select=n_features)
        rfe.fit(X_processed,y_encoded)
        selected_features_rfe=np.array(feature_names_out)[rfe.support_]
        
        st.markdown("Selected Features are:")
        st.code(", ".join(selected_features_rfe))
        x_subset_rfe=x_df[selected_features_rfe]
        
        x_train_rfe,x_test_rfe,y_train_rfe,y_test_rfe=train_test_split(x_subset_rfe,y_encoded,test_size=0.20,random_state=42)

        
        model=get_model(task_type)
        model.fit(x_train_rfe,y_train_rfe)
        st.success("Model is trained for selected column.")
        
        include_target_rfe = st.checkbox("Include target column in heatmap?", value=False, key="checkbox_rfe")
        if include_target_rfe:
            x_subset_rfe["target"] = y_encoded
        corr = x_subset_rfe.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View Feature Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            rfe_score=evaluate_model(model,x_test_rfe,y_test_rfe,task_type)
            st.dataframe(rfe_score)
        
        else:
            selected_report_df,selected_matrix_df,rfe_accuracy=evaluate_model(model,x_test_rfe,y_test_rfe,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(rfe_accuracy * 100):.2f}%")
            
    with tab_forward:
        st.markdown("Forward Feature Selection")
        scoring_metric = 'accuracy' if task_type == 'Classification' else 'r2'
        fwd=SFS(get_model(task_type),k_features=n_features,forward=True,scoring=scoring_metric,cv=5)

        fwd.fit(X_processed,y_encoded)
        selected_indices = list(fwd.k_feature_idx_)
        selected_features_fwd = np.array(feature_names_out)[selected_indices]

        st.markdown("Selected Features are:")
        st.code(", ".join(selected_features_fwd))
        x_subset_fwd=x_df[selected_features_fwd]
        
        x_train_fwd,x_test_fwd,y_train_fwd,y_test_fwd=train_test_split(x_subset_fwd,y_encoded,test_size=0.20,random_state=42)

        
        model=get_model(task_type)
        model.fit(x_train_fwd,y_train_fwd)
        st.success("Model is trained for selected column.")
        
        include_target_forward = st.checkbox("Include target column in heatmap?", value=False, key="checkbox_forward")
        if include_target_forward:
            x_subset_fwd["target"] = y_encoded
        corr = x_subset_fwd.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View Feature Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            fwd_score=evaluate_model(model,x_test_fwd,y_test_fwd,task_type)
            st.dataframe(fwd_score)
        
        else:
            selected_report_df,selected_matrix_df,fwd_accuracy=evaluate_model(model,x_test_fwd,y_test_fwd,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(fwd_accuracy * 100):.2f}%")
    
    with tab_backward:
        st.markdown("BackWard Feature Selection")
        scoring_metric = 'accuracy' if task_type == 'Classification' else 'r2'
        bwd=SFS(get_model(task_type),k_features=n_features,forward=False,scoring=scoring_metric,cv=5)
        bwd.fit(X_processed,y_encoded)
        selected_indices = list(bwd.k_feature_idx_)
        selected_features_bwd = np.array(feature_names_out)[selected_indices]

        st.markdown("Selected Features are")
        st.code(", ".join(selected_features_bwd))
        x_subset_bwd=x_df[selected_features_bwd]
        
        x_train_bwd,x_test_bwd,y_train_bwd,y_test_bwd=train_test_split(x_subset_bwd,y_encoded,test_size=0.20,random_state=42)

        
        model=get_model(task_type)
        model.fit(x_train_bwd,y_train_bwd)
        st.success("Model is trained for selected column.")
        
        include_target_backward = st.checkbox("Include target column in heatmap?", value=False, key="checkbox_backward")
        if include_target_backward:
            x_subset_bwd["target"] = y_encoded
        corr = x_subset_bwd.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View Feature Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            bwd_score=evaluate_model(model,x_test_bwd,y_test_bwd,task_type)
            st.dataframe(bwd_score)
        
        else:
            selected_report_df,selected_matrix_df,bwd_accuracy=evaluate_model(model,x_test_bwd,y_test_bwd,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(bwd_accuracy * 100):.2f}%")
            
    ##########################################################Feature  Extraction ################################################################
    st.subheader("Feature Extraction ")
    n_fe=st.slider("Select the number of Features",1,len(feature_names_out),value=2)
    st.markdown("Select the Feature Extraction method  ")
    
    x_fe=pd.DataFrame(X_processed,columns=feature_names_out)
    y_encoded_new=y_encoded.copy()
    
    pca_tab,tSNE_tab,lda_tab=st.tabs(["PCA","t-SNE","LDA"])
    with pca_tab:
        st.write("Principal Component Analysis")
        
        pca=PCA(n_components=n_fe)
        x_pca=pca.fit_transform(x_fe)
        
        pca_features = [f"PC{i+1}" for i in range(x_pca.shape[1])]
        X_pca_df = pd.DataFrame(x_pca, columns=pca_features)
        y_encoded_new = pd.Series(y_encoded_new).reset_index(drop=True)
        X_pca_df = pd.DataFrame(x_pca, columns=pca_features)
        X_pca_df["Target"] = y_encoded_new
        
        
        with st.expander("View PCA transformed Dataset"):  
            st.dataframe(X_pca_df.head(10))
        
        explained = pca.explained_variance_ratio_

        summary_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained))],
        "Explained Ratio": explained,
        "Cumulative Ratio": np.cumsum(explained)})
        
        with st.expander(" View Component Stats Table"):
                st.dataframe(summary_df)
        
        
        with st.expander("Explained Variance by Component"):
            st.bar_chart(pca.explained_variance_ratio_)
        st.markdown(
        "🔍 **Tip:** Principal components are ranked by how much information they capture.\n"
        "To decide how many to keep, try summing the top values until you reach ~90%.\n"
        "For example, if PC1 + PC2 + PC3 = 92%, keeping those 3 might be ideal."
        )
        

        with st.expander("View cumulative explained variance"):
            st.markdown(
            "🔍 Cumulative variance tells how much total info you've retained as components are added.\n"
            "If the first few components explain most of the variance, you can safely drop the others."
            )
            

            explained_cumsum = np.cumsum(pca.explained_variance_ratio_)
            component_labels = [f"PC{i+1}" for i in range(len(explained_cumsum))]

            fig, ax = plt.subplots()
            ax.bar(component_labels, explained_cumsum, color="skyblue")
            ax.axhline(y=0.9, color="red", linestyle="--", label="90% Cutoff")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Cumulative Variance Explained")
            ax.set_title(" PCA Cumulative Explained Variance")
            ax.legend()

            st.pyplot(fig)
            
        
        
        X_pca_features = X_pca_df.drop("Target", axis=1)
        y_pca_target = X_pca_df["Target"]

        x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_features, y_pca_target, test_size=0.20, random_state=42)        
        model=get_model(task_type)
        model.fit(x_train_pca,y_train_pca)
        st.success("Model is trained for selected column.")
        
        
        corr = X_pca_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View PCA Components Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            pca_score=evaluate_model(model,x_test_pca,y_test_pca,task_type)
            st.dataframe(pca_score)
        
        else:
            selected_report_df,selected_matrix_df,pca_accuracy=evaluate_model(model,x_test_pca,y_test_pca,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(pca_accuracy * 100):.2f}%")
            
    with tSNE_tab:
        st.write("t-distributed Stochastic Neighbor Embedding")
        
        tse=TSNE(n_components=n_fe)
        x_tse=tse.fit_transform(x_fe)
        
        st.write(
        "🔍 t-SNE projects high-dimensional data into 2D or 3D space by preserving **local relationships**.\n"
        "It doesn’t generate measurable components like PCA, so variance charts don’t apply here."
        )
            
        tse_features = [f"Dim{i+1}" for i in range(x_tse.shape[1])]
        X_tse_df = pd.DataFrame(x_tse, columns=tse_features)
        X_tse_df["Target"] = y_encoded_new
        with st.expander("View t-SNE Transformed Dataset"):
            st.dataframe(X_tse_df.head(10))
        
        X_tse_features = X_tse_df.drop("Target", axis=1)
        y_tse_target = X_tse_df["Target"]

        x_train_tse, x_test_tse, y_train_tse, y_test_tse = train_test_split(X_tse_features, y_tse_target, test_size=0.20, random_state=42)        
        model=get_model(task_type)
        model.fit(x_train_tse,y_train_tse)
        st.success("Model is trained for selected column.")
        
        
        corr = X_tse_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.xticks(rotation=45, ha='right',)

        # st.pyplot(fig)
        with st.expander(" View tse transformed Correlation Heatmap"):
            st.pyplot(fig)
    
        if task_type == "Regression":
            tse_score=evaluate_model(model,x_test_tse,y_test_tse,task_type)
            st.dataframe(tse_score)
        
        else:
            selected_report_df,selected_matrix_df,tse_accuracy=evaluate_model(model,x_test_tse,y_test_tse,task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(tse_accuracy * 100):.2f}%")
        
    with lda_tab:
        st.markdown("Linear Discrimination Analysis")  
        if task_type == "Classification":
            max_lda_components = min(x_fe.shape[1], len(np.unique(y_encoded_new)) - 1)
            if n_fe > max_lda_components:
                st.warning(f"⚠️ Selected components exceed LDA's limit. Reducing to {max_lda_components}.")
                n_fe = max_lda_components
                
            lda=LinearDiscriminantAnalysis(n_components=n_fe)
            x_lda=lda.fit_transform(x_fe,y_encoded_new)
            lda_features=[f"LD{i+1}" for i in range(x_lda.shape[1])]
            lda_df=pd.DataFrame(x_lda,columns=lda_features)
            lda_df["Target"]=y_encoded_new
            
            with st.expander(" view LDA transformed Dataset "):
                st.dataframe(lda_df.head(10))
            
            explained = lda.explained_variance_ratio_

            summary_df = pd.DataFrame({
            "Component": [f"LD{i+1}" for i in range(len(explained))],
            "Explained Ratio": explained,
            "Cumulative Ratio": np.cumsum(explained)
})

            with st.expander("View Component Stats Table"):
                st.dataframe(summary_df)
            
            with st.expander("Explained Variance by Component"):
                st.bar_chart(lda.explained_variance_ratio_)
            st.write("**Tip:** LDA components are designed to separate classes. The more variance explained, "
                "the stronger the boundary between groups.")
            
            with st.expander("View cumulative explained variance"):
                st.markdown(
                "🔍 Cumulative variance tells how much total info you've retained as components are added.\n"
                "If the first few components explain most of the variance, you can safely drop the others."
                )
            

                explained_cumsum_lda = np.cumsum(lda.explained_variance_ratio_)
                optimal_index = next((i for i, v in enumerate(explained_cumsum_lda) if v >= 0.9), None)
                if optimal_index is not None:
                    st.success(f"LD1 to LD{optimal_index+1} cover over 90% of class separation.")
            
                component_labels_lda = [f"LD{i+1}" for i in range(len(explained_cumsum_lda))]

                fig, ax = plt.subplots()
                ax.bar(component_labels_lda, explained_cumsum_lda, color="skyblue")
                ax.axhline(y=0.9, color="red", linestyle="--", label="90% Cutoff")
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Cumulative Variance Explained")
                ax.set_title(" LDA Cumulative Explained Variance")
                ax.legend()

                st.pyplot(fig)

            
            X_lda_features = lda_df.drop("Target", axis=1)
            y_lda_target = lda_df["Target"]

            x_train_lda, x_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda_features, y_lda_target, test_size=0.20, random_state=42)        
            model=get_model(task_type)
            model.fit(x_train_lda,y_train_lda)
            st.success("Model is trained for selected column.")
        
        
            corr = lda_df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            plt.xticks(rotation=45, ha='right',)

            # st.pyplot(fig)
            with st.expander(" View LDA Components Correlation Heatmap"):
                st.pyplot(fig)
                
            selected_report_df, selected_matrix_df, lda_accuracy = evaluate_model(model, x_test_lda, y_test_lda, task_type)
            tab1,tab2=st.tabs(["Classification Report","Confusion Matrix"])
            with tab1:
                st.dataframe(selected_report_df,hide_index=True)
            with tab2:
                st.dataframe(selected_matrix_df)
            
            st.metric("Accuracy", f"{(lda_accuracy * 100):.2f}%")
            
            
            
        else:
            st.warning("LDA is only available for classification tasks.")
        
        
    st.subheader("Classification Technique Comparison")
    
    if task_type=="Classification":
        accuracy_dict = {
            "Default": default_accuracy,
            "Manual": manual_accuracy,
            "Top-N": topN_accuracy,
            "RFE": rfe_accuracy,
            "Forward": fwd_accuracy,
            "Backward": bwd_accuracy,
            "PCA": pca_accuracy,
            "t-SNE": tse_accuracy,
            "LDA": lda_accuracy
        }

        accuracy_df = pd.DataFrame({
            "Technique": list(accuracy_dict.keys()),
            "Accuracy": list(accuracy_dict.values())
        })


        fig = px.bar(accuracy_df,x="Technique",y="Accuracy",text=accuracy_df["Accuracy"].apply(lambda x: f"{x:.2%}"),opacity=0.8,title="Accuracy by Technique",color_discrete_sequence=["purple"])


        fig.update_layout(
        yaxis_title="Accuracy",
        xaxis_title="Technique",
        template="simple_white",
        uniformtext_minsize=10,
        uniformtext_mode='hide'
        )

        st.plotly_chart(fig, use_container_width=True)

    
    else :
        scores_dict = {
            "Default": default_score,
            "MANUAL": manual_score,
            "TOP N": topN_score,
            "RFE": rfe_score,
            "FORWARD": fwd_score,
            "BACKWARD": bwd_score,
            "PCA": pca_score,
            "TSE": tse_score,
        }
        technique_order = list(scores_dict.keys())


        combined_df = pd.concat([
        df.assign(Technique=name) for name, df in scores_dict.items()
        ], ignore_index=True)[["Technique", "Metric", "Score"]]

        wide_df = combined_df.pivot(index="Technique", columns="Metric", values="Score").reset_index()
        wide_df.columns.name = None
        wide_df.rename(columns={"R2": "R² Score"}, inplace=True)
        wide_df["MSE"] = wide_df["MSE"].round(3)
        wide_df["R² Score"] = wide_df["R² Score"].round(2)

        wide_df["Technique"] = pd.Categorical(wide_df["Technique"], categories=technique_order, ordered=True)
        wide_df = wide_df.sort_values("Technique")
        st.dataframe(wide_df.style.format({"Score": "{:.2f}"}))
        


        fig = go.Figure(data=[
        go.Bar(name='MSE', x=wide_df['Technique'], y=wide_df['MSE'], marker_color='indianred'),
        go.Bar(name='R² Score', x=wide_df['Technique'], y=wide_df['R² Score'], marker_color='steelblue')
        ])

        fig.update_layout(
        title="Comparison of Feature Selection Techniques",
        xaxis_title="Technique",
        yaxis_title="Score",
        barmode='group',
        height=500,
        width=900,
        template='plotly_white'
)


        st.plotly_chart(fig, use_container_width=True)



        



else:
    placeholder = st.empty()
    with placeholder:
        st.warning(" Select valid Dataset")
    time.sleep(3)
    placeholder.empty()





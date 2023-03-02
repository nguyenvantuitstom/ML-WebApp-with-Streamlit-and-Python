# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import streamlit.components.v1 as components
from datetime import date, timedelta

import yfinance as yf
from plotly import graph_objs as go
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sns
import altair as alt
from altair.utils.data import to_values
Page_config = {"page_title": "Stock Forecast", "layout": "wide", "initial_sidebar_state": "auto"}
st.set_page_config(**Page_config)
# TODAY = date.today().strftime("%Y-%m-%d")
bins = []
maxSigma = 3
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Trade Zooms: Stock Forecast")


def plot_raw_data(data, rawData, INDICATORS):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'],
                  y=data['close'], name="Close Price"))
    for col in rawData.columns:
        if col.upper() in INDICATORS:
            fig.add_trace(go.Scatter(
                x=rawData['date'], y=rawData[col], name=col))
    fig.layout.update(
        title_text='Discover how your strategy consistent with price movement(expander... ROC, RME, Heat Map)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# """
# Next we clean our data and perform feature engineering to create new technical indicator features that our
# model can learn from
# """


def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """

    return data.ewm(alpha=alpha).mean()


def create_n_steps(close, Distance, n):
    n_steps_upper = np.linspace(close + Distance, close, num=n, endpoint=False)
    n_steps_lower = np.linspace(close - Distance, close, num=n, endpoint=False)
    n_steps = np.concatenate((n_steps_upper, [close], n_steps_lower[::-1]))
    n_steps = np.sort(n_steps)
    return n_steps[~np.isnan(n_steps)]


def my_cut(number, bins, labels=None):
    if len(bins) > len(labels):
        # print(bins)
        # print(labels)
        result = pd.cut([number], bins=bins, labels=labels)
        if number < bins[0]:
            return 0
        if number > bins[len(bins) - 1]:
            return len(bins) - 2
        return result[0]
    return n_steps_const/2


def label(row):
    global n_steps_const
    tmpSteps = n_steps_const
    if tmpSteps%2 != 0:
        n_steps_const = n_steps_const + 1
    n_steps = create_n_steps(
        close=row['close'], Distance=row['Distance'], n=int(n_steps_const/2))
    labels_nac = [f"{i}" for i in list(range(0, n_steps_const))]
    if np.isnan(row['close_shift1']):
        return int(n_steps_const/2)
    label = my_cut(row['close_shift1'], bins=list(n_steps), labels=labels_nac)
    return int(label)


def _get_indicator_data(data, INDICATORS):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    for indicator in INDICATORS:
        if "EMA9" in indicator:
            data['EMA9'] = data['close'] / data['close'].ewm(9).mean()
        elif "EMA34" in indicator:
            data['EMA34'] = data['close'] / data['close'].ewm(34).mean()
        elif "EMA89" in indicator:
            data['EMA89'] = data['close'] / data['close'].ewm(89).mean()
        else:
            ind_data = eval('TA.' + indicator + '(data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            data = data.merge(ind_data, left_index=True, right_index=True)
            data.rename(
                columns={"14 period " + indicator: indicator}, inplace=True)
            data.rename(columns={"14 period " + indicator +
                        " %K": indicator}, inplace=True)

    if len(INDICATORS) > 0:
        data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    return data


def _decileData(data):
    # res, qbins = pd.qcut(data['Body'], 5, retbins=True)
    qbins = np.quantile(data['Body'], q=np.arange(0.20, 1, 0.2))
    binsize = abs((qbins[len(qbins) - 1] - qbins[0])/len(qbins)/100)
    print(qbins)
    print("binsize")
    print(binsize)
    data['Distance'] = n_steps_const/2*data['close']*binsize

    data['Class'] = data[['close', 'close_shift1', 'Distance']].apply(
        label, axis=1)

    bins = create_n_steps(close=0, Distance=binsize *
                          100, n=int(n_steps_const/2))
    data['Trend'] = data['close_shift1'] > data['close']

    scale = StandardScaler()
    y = data[['Class', 'Trend']]
    scaledDataY = scale.fit_transform(y)
    scaledDataY = np.clip(scaledDataY, a_max=maxSigma, a_min=-maxSigma)
    scaledDataY = scale.inverse_transform(scaledDataY)
    data['Class'] = scaledDataY[:, 0]

    return data, bins


def _produce_prediction(data, window):
    # data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()
    body = data['open'] - data['close']
    change = (data['close'] - data['open'])/data['open']*100
    data['Body'] = change
    return data


def getDisplayFields(data):
    return [x for x in data.columns if x not in ['Adj Close', 'normVol', 'close_shift1', 'Trend', 'Class', 'Body', 'Distance', 'volume']]


def getIgnoreFields():
    return ['Adj Close', 'date', 'Class', 'Trend', 'close_shift1', 'Distance']

def load_data(ticker, START, TODAY):
        # pd.options.display.float_format = '${:,.2f}'.format
        startStr = START.strftime("%Y-%m-%d")
        endStr = TODAY.strftime("%Y-%m-%d")
        data = yf.download(ticker, START, TODAY + timedelta(days=1))
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%m/%d/%Y')
        return data

def plot_metrics(metrics_list, model, X_test, y_test, class_names):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        # plot_confusion_matrix(model, X_test, y_test, display_labels = class_names)
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        # plot_roc_curve(model, X_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        # plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()

def main():
    
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock, START, TODAY)

    data_load_state.text('')
    data['close_shift1'] = data['Close'].shift(-1)
    rawData = data
    data.rename(columns={"Date": 'date', "Close": 'close', "High": 'high',
            "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)

    data = _get_indicator_data(data, INDICATORS)
    data = _produce_prediction(data, window=15)

    data, bins = _decileData(data)
    displayData = data[getDisplayFields(data)]
    st.write(displayData.tail())

    # st.bar_chart(data['Decile'].value_counts())
    class_labels = np.unique(data['Class'].values)

    print("bins")
    print(bins)

    rawData = data
    # Some indicators produce NaN values for the first few rows, we just remove them here
    data = data.dropna()

    # Split data into equal partitions of size len_train
    # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    num_train = 10
    len_train = 40  # Length of each train-test set

    # Models which will be used
    mlp = MLPClassifier()
    # rf = RandomForestClassifier()
    rf = MLPClassifier()
    knn = KNeighborsClassifier()

    # Create a tuple list of our models
    estimators = [('knn', knn), ('rf', rf)]
    ensemble = VotingClassifier(estimators, voting='soft')

    # Predict trend
    y_trend = data['Trend']
    features_trend = [x for x in data.columns if x not in getIgnoreFields()]
    X_trend = data[features_trend]

    scaleTrend = StandardScaler()
    scaled_Body_Trend = scaleTrend.fit_transform(X_trend)
    scaledDataTrend = np.clip(
        scaled_Body_Trend, a_max=maxSigma, a_min=-maxSigma)
    X_trend = scaledDataTrend

    # """
    # Trend labels
    # """
    # st.bar_chart(data['Trend'].value_counts())

    X_train_trend, X_test_trend, y_train_trend, y_test_trend = train_test_split(
        X_trend, y_trend, test_size=0.3, random_state=42)
    mlp.fit(X_train_trend, y_train_trend)
    mlp_prediction = mlp.predict(X_test_trend)
    predictResult_trend = int(mlp_prediction[len(mlp_prediction) - 1])
    scores_trend = mlp.predict_proba(X_test_trend)

    st.bar_chart(data['Class'].value_counts())
    # Filter data
    data = data[data['Trend'] == predictResult_trend]
    # st.write(data)
    # Predict price volatility
    y = data['Class']

    features = [x for x in data.columns if x not in getIgnoreFields()]
    X = data[features]
    # st.write(X)

    scale = StandardScaler()
    scaled_Body = scale.fit_transform(X)
    scaledData = np.clip(scaled_Body, a_max=maxSigma, a_min=-maxSigma)
    X = scaledData
    # st.write(scale.inverse_transform(scaledData))
    # fit models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    # get predictions
    rf_prediction = rf.predict(X_test)
    knn_prediction = knn.predict(X_test)
    ensemble_prediction = ensemble.predict(X_test)

    # print(rf_prediction)
    # determine accuracy and append to results
    rf_accuracy = accuracy_score(y_test.values, rf_prediction)
    knn_accuracy = accuracy_score(y_test.values, knn_prediction)
    ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)
    mlp_accuracy = accuracy_score(y_test_trend.values, mlp_prediction)

    predictResult = int(rf_prediction[len(rf_prediction) - 1])

    class_accuracy = {}
    class_accuracy_display = {}
    class_current_accuracy_display = {}
    sortedLabels = (sorted(class_labels, reverse=False))
    scores = rf.predict_proba(X_test)

    print("predictResult_trend")
    print(predictResult_trend)
    print("scores_trend")
    print(scores_trend)

    detailFormatNumber = "{:,.2f}"
    normalFormatNumber = "{:,.1f}"

    fromNum = round(bins[predictResult], 2)
    toNum = round(bins[predictResult + 1], 2)
    fromValue = detailFormatNumber.format(fromNum) + "%"
    toValue = detailFormatNumber.format(toNum) + "%"
    closePrice = rawData['close'].iloc[-1]
    fromPrice = (closePrice*fromNum/100)+closePrice
    toPrice = (closePrice*toNum/100)+closePrice

    trendType = 0
    predictScore = scores[len(scores) - 1]
    predictScore_trend = scores_trend[len(scores_trend) - 1]

    probability = 0.0
    predictResultIndex = 0
    maxPredicLabel = 0.0
    maxPredicValue = 0.0

    currentDisplayLabels = []
    currentDisplayValues = []
    displayLabels = []
    displayValues = []

    currentBinIndex = 0
    binPos = 0.0
    binNav = 0.0
    predictTrend_Score = 0.0

    probableVolatility = 0.0
    binIndexBalance = len(bins)/2
    while currentBinIndex < len(bins):
        num = round(bins[currentBinIndex], 2)
        if num < 0:
            binNav = binNav + \
                round(bins[currentBinIndex], 2) * \
                predictScore[int(currentBinIndex)]
        else:
            if num == 0:
                binIndexBalance = currentBinIndex
            else:
                binPos = binPos + \
                    round(bins[currentBinIndex], 2) * \
                          predictScore[int(currentBinIndex) %
                                           int(len(bins)/2) - 1]
        currentBinIndex = currentBinIndex + 1

    contentTrend = ""
    if predictResult_trend == 0:
        probableVolatility = bins[binIndexBalance - 1]
        contentTrend = "or lower"
    else:
        probableVolatility = bins[binIndexBalance + 1]
        contentTrend = "or higher"
    weightedValue = (binNav if predictResult_trend == 0 else binPos)

    # """
    # Decile Labels distribution
    # """
    print("Decile")
    print(data['Class'].value_counts())
    # st.bar_chart(data['Decile'].value_counts())

    # st.subheader("Bins %" + str(bins))
    st.header("Backtest result:")
    st.subheader("Today prediction")
    st.text("Most probable volatility: " +
            detailFormatNumber.format(probableVolatility) + "%" + " " + contentTrend)
    st.text("Weighted volatility:" + " " +
            detailFormatNumber.format(weightedValue) + "%")
    st.text("Price movement prediction: " + detailFormatNumber.format((toPrice if predictResult_trend == 0 else fromPrice)
                                                                      ) + " -> " + detailFormatNumber.format((fromPrice if predictResult_trend == 0 else toPrice)) + " " + contentTrend)
    # st.caption(content + " with probability " + "{:.2f}".format(probability*100) + "%")
    # st.caption("Price range: " + fromValue + " -> " + toValue)
    print(sortedLabels)

    print("rf_prediction")
    print(rf_prediction)

    print("close")
    close = rawData['close'].iloc[-1]
    print(close)

    print("predictScore")
    print(predictScore)
    dictScore = {}
    class_result_labels = np.unique(y_test.values)
    for idx, label in enumerate(class_result_labels):
        dictScore[label] = predictScore[idx]

    print(dictScore)
    for label in sortedLabels:
        idx = np.where(y_test == label)
        # class_accuracy.append({"label": str(label), "value": str(accuracy_score(y_test.values[idx], rf_prediction[idx]))})
        class_accuracy[label] = accuracy_score(
            y_test.values[idx], rf_prediction[idx])
        fromValue = "0.0"
        toValue = "0.0"
        index = int(label)

        if predictResult_trend == 0:
            fromValue = normalFormatNumber.format(round(bins[index + 1], 2))
            toValue = normalFormatNumber.format(round(bins[index], 2))
        else:
            fromValue = normalFormatNumber.format(round(bins[index], 2))
            toValue = normalFormatNumber.format(round(bins[index + 1], 2))

        value = accuracy_score(y_test.values[idx], rf_prediction[idx])
        if np.isnan(value):
            value = 0.1/100
        if maxPredicValue < value:
            maxPredicValue = value
            maxPredicLabel = label

        strToValue = detailFormatNumber.format(
            (float(toValue)/100*close + close))

        print("valuevaluevalue")
        print(value)
        class_accuracy_display[strToValue] = value
        displayLabels.append(strToValue)
        displayValues.append(value*100)

        probabilityValue = 0.1
        if label in dictScore:
            probabilityValue = dictScore[label]*100

        class_current_accuracy_display[strToValue] = value
        currentDisplayLabels.append(strToValue)
        currentDisplayValues.append(probabilityValue)
        # print(class_accuracy_display)

    print("class_current_accuracy_display")
    print(class_current_accuracy_display)

    # st.markdown("Volatility probability:")
    # st.bar_chart(data=class_current_accuracy_display)
    # rf.predict_proba(X_test)

    chart_data = pd.DataFrame({
        'Probability %': currentDisplayValues,
        ' ': currentDisplayLabels,
        'variable': range(len(currentDisplayLabels))
    })
    # currentData = pd.melt(chart_data.reset_index(), id_vars=["index"])

    col1, col2 = st.columns(2)
    with col1:
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x='Probability %',
                y=alt.Y(' ', sort=alt.EncodingSortField(
                    ' ', op='min', order='descending')),
                # alt.Y('volatility', sort=alt.EncodingSortField('probability', op='min', order='descending')),
                # order=alt.Order("variable", sort="ascending"),
            ).properties(
                height=60*n_steps_const
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.area_chart(data["close"], use_container_width=True)

    maxFromNum = round(bins[int(maxPredicLabel)], 2)
    maxFromValue = detailFormatNumber.format(maxFromNum) + "%"

    maxToNum = round(bins[int(maxPredicLabel) + 1], 2)
    maxToValue = detailFormatNumber.format(maxToNum) + "%"

    # cumWeightedValue = ((maxPredicValue * maxFromNum) + (maxPredicValue * maxToNum))

    maxPredicLabel_trend = predictResult_trend
    maxPredicValue_trend = 0.0
    class_labels_trend = np.unique(y_test_trend.values)
    sortedLabels_trend = (sorted(class_labels_trend, reverse=False))
    idx = np.where(y_test_trend == predictResult_trend)
    maxPredicValue_trend = accuracy_score(
        y_test_trend.values[idx], mlp_prediction[idx])

    maxContentTrend = "or higher"
    if maxPredicLabel_trend == 0:
        maxContentTrend = "or lower"

    cumBinNav = 0.0
    cumBinPos = 0.0
    cumCurrentBinIndex = 0
    while cumCurrentBinIndex < len(bins):
        num = round(bins[cumCurrentBinIndex], 2)
        if num < 0:
            cumBinNav = cumBinNav + \
                round(bins[cumCurrentBinIndex], 2) * \
                displayValues[int(cumCurrentBinIndex)]/100
        else:
            if num == 0:
                binIndexBalance = cumCurrentBinIndex
            else:
                print(
                    "displayValues[int(cumCurrentBinIndex)%int(len(bins)/2) - 1]")
                print(round(bins[cumCurrentBinIndex], 2))
                print(displayValues[int(cumCurrentBinIndex) %
                      int(len(bins)/2) - 1])
                cumBinPos = cumBinPos + round(bins[cumCurrentBinIndex], 2)*displayValues[int(
                    cumCurrentBinIndex) % int(len(bins)/2) - 1]/100
        cumCurrentBinIndex = cumCurrentBinIndex + 1

    cumWeightedValue = (cumBinNav if maxPredicLabel_trend == 0 else cumBinPos)
    cumProableVolatility = 0.0
    if maxPredicLabel_trend == 0:
        cumProableVolatility = bins[binIndexBalance - 1]
    else:
        cumProableVolatility = bins[binIndexBalance + 1]

    st.subheader("History backtest summary" +
                 "(" + str((TODAY - START).days) + "days" + ")")
    st.text("Most probable volatility:" + " " +
            detailFormatNumber.format(cumProableVolatility) + "%" + " " + maxContentTrend)
    st.text("Weighted volatility:" + " " +
            detailFormatNumber.format(cumWeightedValue) + "%")
    st.text("Indicators vs price change consistency:" + " " +
            normalFormatNumber.format(mlp_accuracy*100) + "%" + " Accuracy")
    # st.bar_chart(data=class_accuracy_display)

    chart_data = pd.DataFrame({
        'Probability %': displayValues,
        ' ': displayLabels,
        'variable': range(len(displayLabels))
    })
    # currentData = pd.melt(chart_data.reset_index(), id_vars=["index"])

    print('RF Accuracy = ' + str(rf_accuracy))
    print('KNN Accuracy = ' + str(knn_accuracy))
    print('ENSEMBLE Accuracy = ' + str(ensemble_accuracy))
    print('MLP Accuracy = ' + str(mlp_accuracy))
    # print('Accuracy score: ', accuracy_score(y_test.values, mlp_prediction))
    # print('Confusion matrix: \n', confusion_matrix(y_test.values, mlp_prediction))
    print('Classification MLP report: \n', classification_report(
        y_test_trend.values, mlp_prediction))

    # with st.expander("See explanation"):
    #     st.write("The chart above shows some numbers I picked for you. I rolled actual dice for these, so they're *guaranteed* to be random.")
    #     st.image("https://static.streamlit.io/examples/dice.jpg")
    with st.expander("See explanation"):
        class_names = class_result_labels
        st.subheader("Choose Classifier")
        classifier = st.selectbox("Classifier", ("Random Forest", "Support Vector Machine (SVM)", "Logistic Regression"))
    
        if classifier == "Support Vector Machine (SVM)":
            st.subheader("Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
            kernel = st.radio("Kernel", ("rbf", "linear"), key = 'kernel')
            gamma = st.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'auto')
        
            # metrics = st.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            metrics = 'Confusion Matrix'
            if st.button("Classify", key = 'classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C = C, kernel = kernel, gamma = gamma)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                plot_metrics(metrics, model, X_test, y_test, class_names)

        if classifier == "Logistic Regression":
            st.subheader("Model Hyperparameters")
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
            max_iter = st.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
            
            # metrics = st.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            metrics = 'Confusion Matrix'
            if st.button("Classify", key = 'classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C = C, max_iter = max_iter)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                plot_metrics(metrics, model, X_test, y_test, class_names)

        if classifier == "Random Forest":
            st.subheader("Model Hyperparameters")        
            n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
            max_depth = st.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
            bootstrap = st.radio("Bootstrap samples when building trees", (True, False), key = 'bootstrap')
            # metrics = st.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            metrics = 'Confusion Matrix'
            if st.button("Classify", key = 'classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names, average='micro').round(2))
                plot_metrics(metrics, model, X_test, y_test, class_names)
                            
    # chart = (
    #     alt.Chart(chart_data)
    #     .mark_bar()
    #     .encode(
    #         x='Probability %',
    #         y=alt.Y(' ', sort=alt.EncodingSortField(
    #             ' ', op='min', order='descending')),
    #         # alt.Y('volatility', sort=alt.EncodingSortField('probability', op='min', order='descending')),
    #         # order=alt.Order("variable", sort="ascending"),
    #     ).properties(
    #         height=300
    #     )
    # )

    # st.altair_chart(chart, use_container_width=True)
    # plot_raw_data(data, rawData, INDICATORS)

components.html(
     """
        <a href=https://tradezooms.net/ target="_blank">Home Page</a>

        <style>
        a:link, a:visited {
        background-color: #f44336;
        color: white;
        padding: 15px 25px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        }

        a:hover, a:active {
          background-color: red;
        }
        </style>

        <a href=https://tradezooms.net/cong-cu-giao-dich/ target="_blank">Free Trading Tool for MT4</a>

        """,
      height=60
     )
menu = ["Home", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)
if choice == "Home":
    with st.form("my_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col4:
            n_steps_const = st.number_input("Bin number",
                        min_value=2,
                        max_value=20,
                        value=6,
                        step=2)
            # st.write(n_steps_const)
        with col1:
            stocks = ('ETH-USD', 'AAPL', 'MSFT', 'GME')
            selected_stock = st.selectbox('Select dataset for prediction:', stocks)
        with col2:
            START = st.date_input(
                    "Start",
                    datetime.date(2022, 1, 1))
        with col3:
            TODAY = st.date_input(
                    "End",
                    date.today())
        INDICATORS = st.multiselect(
                    'Select indicators:',
                    ['RSI', 'MACD', 'STOCH', 'ATR', 'MOM', 'EMA9', 'EMA34', 'EMA89',
                        'MFI', 'ADL', 'ROC', 'OBV', 'CCI', 'EMV'],
                ['MOM']
            )
        submitted = st.form_submit_button("Calculate Now")
            
    if submitted:
        data = load_data(selected_stock, START, TODAY)
main()
components.html(
    """
    <a href=https://tradezooms.net/cong-cu-giao-dich/ target="_blank">Come back to Home Page</a>

    <style>
    a:link, a:visited {
    background-color: #f44336;
    color: white;
    padding: 15px 25px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    }

    a:hover, a:active {
      background-color: red;
    }
    </style>

    """,
    height=60
)


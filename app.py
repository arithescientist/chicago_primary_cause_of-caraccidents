import flask
from flask import request, render_template 
import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from flask_cors import CORS, cross_origin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder


lb = MultiLabelBinarizer()
scaler = StandardScaler(with_mean=False)
app = flask.Flask(__name__)
app.config["DEBUG"] = True

CORS(app)

# main index page route
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


def feature_engine(data):
    row = data[0]

    transform_data = pd.DataFrame(row)
    transform_data.columns = ['SPEED LIMIT', 'AGE','DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
                                'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']
    
    non_normal = ['SPEED LIMIT', 'AGE']
    
    data_cat = transform_data[['DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
                                'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']].astype('category')
    data_cont = transform_data[non_normal].astype(float)

    return data_cont,data_cat
    
 
def test_multilabel(classifier, X_test):
    CT = ['DRIVER VISION','DRIVER ACTION','PHYSICAL CONDITION',
                                'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']

    input = pd.get_dummies(X_test, columns = CT, drop_first = True).toarray()
  

    input = scaler.fit_transform(input)

    #input = np.ravel(input)
    
    input = np.array(input).reshape(1,-1)

    classifier.fit(input.T)
    predictions = classifier.predict(input.T)
    predictions = predictions[0,:].tolist()
    target = ['FOLLOWING TOO CLOSELY', 'FAILING TO YIELD RIGHTOFWAY',
               'IMPROPER BACKING', 'SPEED RELATED', 'WEATHER',
               'DISREGARDING TRAFFIC SIGNALS', 'DISTRACTION  FROM INSIDE VEHICLE',
               'INTOXICATED/PHYSICAL CONDITION', 'IMPROPER LANE USAGE',
               'DISTRACTION  FROM OUTSIDE VEHICLE', 'IMPROPER TURNINGNO SIGNAL',
               'EQUIPMENT  VEHICLE CONDITION',
               'DRIVING SKILLSKNOWLEDGEEXPERIENCE', 'IMPROPER OVERTAKINGPASSING',
               'ROAD CONDITIONS']
    pred = []
    for i in range(len(predictions)):
        if predictions[i]==1:
            pred.append(target[i])
            output_result = "The cause of the accident is the: {}".format(pred)
    return output_result    
    
def model_predict(x, model):
   
    CT = ['DRIVER VISION','DRIVER ACTION','PHYSICAL CONDITION',
                                 'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']

    # input = pd.get_dummies(x, columns = CT, drop_first = True)
    
    preds = lb.fit_transform(x[CT])


    preds = scaler.fit_transform(preds)


    
    preds = np.array(preds).reshape(-1,1)



    preds = model.predict_proba(preds)
    
    preds = np.ravel(preds)
    preds = np.argpartition(preds, -1)[-1:]

    #preds = np.argsort(preds, axis=1)[:,-3:]
    def largest_indices(ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    

    preds = model.classes_[largest_indices(preds, 1)] 
    return preds        
   


@app.route('/upload')
def upload_file2():
    return render_template('index.html', title='Home')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        # Get the file from post request
        
        speed = int(request.args['SPEED LIMIT'])
        age = int(request.args['AGE'])
        vision = request.args['DRIVER VISION']
        action = request.args['DRIVER ACTION']
        physical = request.args['PHYSICAL CONDITION']
        roadway = request.args['ROADWAY CONDITION']
        device = request.args['DEVICE CONDITION']
        fcrash = request.args['FIRST CRASH TYPE']
        filename = 'xgb.sav'
        #serInput = [speed,age,vision,action,physical,roadway,device,fcrash]
        model = pickle.load(open(filename, 'rb'))
        data = {'SPEED LIMIT':[speed],'AGE':[age],'DRIVER VISION':[vision],'DRIVER ACTION':[action],
         'PHYSICAL CONDITION':[physical],'ROADWAY CONDITION':[roadway],'DEVICE CONDITION':[device],'FIRST CRASH TYPE':[fcrash]}

        input = pd.DataFrame(data, columns = ['SPEED LIMIT','AGE','DRIVER VISION','DRIVER ACTION','PHYSICAL CONDITION',
                                 'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE'])
         
        preds = model_predict(input, model)

 


        #preds =str(preds.idxmax(axis=1)[0])
#########################workimg
        # label=['FOLLOWING TOO CLOSELY','FAILING TO YIELD RIGHTOFWAY',
        #        'IMPROPER BACKING', 'SPEED RELATED', 'WEATHER',
        #        'DISREGARDING TRAFFIC SIGNALS', 'DISTRACTION  FROM INSIDE VEHICLE',
        #        'INTOXICATED/PHYSICAL CONDITION', 'IMPROPER LANE USAGE',
        #        'DISTRACTION  FROM OUTSIDE VEHICLE', 'IMPROPER TURNINGNO SIGNAL',
        #        'EQUIPMENT  VEHICLE CONDITION',
        #        'DRIVING SKILLSKNOWLEDGEEXPERIENCE', 'IMPROPER OVERTAKINGPASSING',
        #        'ROAD CONDITIONS']
    
        # pred_name = label[np.argmax(preds)]
##################################
    

        return render_template('results.html', predictions=preds)
    return None


# @app.route('/upload',methods=['GET'])
# @cross_origin()
# def predict():
#     if request.method == 'GET':
        
#         #  reading the inputs given by the user
#         speed = int(request.args['SPEED LIMIT'])
#         age = int(request.args['AGE'])
#         vision = request.args['DRIVER VISION']
#         action = request.args['DRIVER ACTION']
#         physical = request.args['PHYSICAL CONDITION']
#         roadway = request.args['ROADWAY CONDITION']
#         device = request.args['DEVICE CONDITION']
#         fcrash = request.args['FIRST CRASH TYPE']
        
#         userInput = [[speed,age,vision,action,physical,roadway,device,fcrash]]
        
        
#         model =  pickle.load(open('xgb1.pickle', 'rb')) # need to use as training model is onehot encoded
        
#         feature_selection=PCA()
#         model = make_pipeline_imb
# #  make a database
# # asign a variable for cat
# #  get onehotencoding
# # 
#         input = pd.DataFrame(userInput)
#         input.columns = ['SPEED LIMIT', 'AGE','DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
#                                 'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']
#         CT = ['DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
#                                 'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']

#         input = pd.get_dummies(input, columns = CT, drop_first = True)
       
#         scaler.fit(input)
#         input = scaler.transform(input)

#         input = np.ravel(input)
        
#         input = np.array(input).reshape(1,-1)

#         preds = model.predict(input.T)
#         label=['FOLLOWING TOO CLOSELY','FAILING TO YIELD RIGHTOFWAY',
#        'IMPROPER BACKING', 'SPEED RELATED', 'WEATHER',
#        'DISREGARDING TRAFFIC SIGNALS', 'DISTRACTION  FROM INSIDE VEHICLE',
#        'INTOXICATED/PHYSICAL CONDITION', 'IMPROPER LANE USAGE',
#        'DISTRACTION  FROM OUTSIDE VEHICLE', 'IMPROPER TURNINGNO SIGNAL',
#        'EQUIPMENT  VEHICLE CONDITION',
#        'DRIVING SKILLSKNOWLEDGEEXPERIENCE', 'IMPROPER OVERTAKINGPASSING',
#        'ROAD CONDITIONS']
    
#         pred_name = label[np.argmax(preds)]
#         probability = preds[np.argmax(preds)]

#         return render_template('results.html', prediction= probability)




if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(host='0.0.0.0', debug=True) # running the app
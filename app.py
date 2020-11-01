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


lb = OneHotEncoder()
app = flask.Flask(__name__)
app.config["DEBUG"] = True
numerical = ['SPEED LIMIT', 'AGE']
categorical = ['DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
                                'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']

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
    preds = pd.get_dummies(x, columns = CT, drop_first = True)


    fname = 'scaler.sav'
    scaler = pickle.load(open(fname, 'rb'))
    
    scaler.fit(preds)
    preds = scaler.transform(preds)

    filename = 'xgb.sav'
    model = pickle.load(open(filename, 'rb'))

    preds = np.array(preds)
    

    # preds = np.array(preds).reshape(1,-1)


    # preds = np.ravel(preds)
    preds = model.predict(preds)[0]
    
    
    # preds = np.argpartition(preds, -1)[-1:]

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

def analysis(speed, age, vision, action, physical, roadway, device, fcrash):
    predictions = []
    recommend_index = []
    recommend = []
    column = ['SPEED LIMIT', 'AGE','DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
                                'ROADWAY CONDITION','DEVICE CONDITION','FIRST CRASH TYPE']
    
    serInput = [speed,age,vision,action,physical,roadway,device,fcrash]
    
    data = pd.DataFrame([serInput], columns=column)
    
    df_num = data[numerical]
    df_cat = data[categorical]
    #df_cat.columns = [''] * len(df_cat.columns)
    df_cat = pd.get_dummies(df_cat)
    #df_cat.columns = df_cat.columns.str[1:]
    df_missing = pd.concat([df_num, df_cat], axis=1)
    


    filename = 'xgb.sav'
    model = pickle.load(open(filename, 'rb'))

    cols_when_model_builds = ['SPEED LIMIT', 'AGE',
       'DRIVER VISION_BLINDED - SUNLIGHT', 'DRIVER VISION_BLOWING MATERIALS',
       'DRIVER VISION_BUILDINGS', 'DRIVER VISION_EMBANKMENT',
       'DRIVER VISION_HILLCREST', 'DRIVER VISION_MOVING VEHICLES',
       'DRIVER VISION_NOT OBSCURED', 'DRIVER VISION_OTHER',
       'DRIVER VISION_PARKED VEHICLES', 'DRIVER VISION_SIGNBOARD',
       'DRIVER VISION_TREES, PLANTS', 'DRIVER VISION_UNKNOWN',
       'DRIVER VISION_WINDSHIELD (WATER/ICE)',
       'DRIVER ACTION_DISREGARDED CONTROL DEVICES',
       'DRIVER ACTION_EMERGENCY VEHICLE ON CALL',
       'DRIVER ACTION_EVADING POLICE VEHICLE', 'DRIVER ACTION_FAILED TO YIELD',
       'DRIVER ACTION_FOLLOWED TOO CLOSELY', 'DRIVER ACTION_IMPROPER BACKING',
       'DRIVER ACTION_IMPROPER LANE CHANGE', 'DRIVER ACTION_IMPROPER PARKING',
       'DRIVER ACTION_IMPROPER PASSING', 'DRIVER ACTION_IMPROPER TURN',
       'DRIVER ACTION_LICENSE RESTRICTIONS', 'DRIVER ACTION_NONE',
       'DRIVER ACTION_OTHER', 'DRIVER ACTION_OVERCORRECTED',
       'DRIVER ACTION_STOPPED SCHOOL BUS', 'DRIVER ACTION_TEXTING',
       'DRIVER ACTION_TOO FAST FOR CONDITIONS', 'DRIVER ACTION_UNKNOWN',
       'DRIVER ACTION_WRONG WAY/SIDE', 'PHYSICAL CONDITION_FATIGUED/ASLEEP',
       'PHYSICAL CONDITION_HAD BEEN DRINKING',
       'PHYSICAL CONDITION_ILLNESS/FAINTED',
       'PHYSICAL CONDITION_IMPAIRED - ALCOHOL',
       'PHYSICAL CONDITION_IMPAIRED - ALCOHOL AND DRUGS',
       'PHYSICAL CONDITION_IMPAIRED - DRUGS', 'PHYSICAL CONDITION_MEDICATED',
       'PHYSICAL CONDITION_NORMAL', 'PHYSICAL CONDITION_OTHER',
       'PHYSICAL CONDITION_REMOVED BY EMS', 'PHYSICAL CONDITION_UNKNOWN',
       'ROADWAY CONDITION_ICE', 'ROADWAY CONDITION_OTHER',
       'ROADWAY CONDITION_SAND, MUD, DIRT', 'ROADWAY CONDITION_SNOW OR SLUSH',
       'ROADWAY CONDITION_UNKNOWN', 'ROADWAY CONDITION_WET',
       'DEVICE CONDITION_FUNCTIONING PROPERLY', 'DEVICE CONDITION_MISSING',
       'DEVICE CONDITION_NO CONTROLS', 'DEVICE CONDITION_NOT FUNCTIONING',
       'DEVICE CONDITION_OTHER', 'DEVICE CONDITION_UNKNOWN',
       'DEVICE CONDITION_WORN REFLECTIVE MATERIAL', 'FIRST CRASH TYPE_ANIMAL',
       'FIRST CRASH TYPE_FIXED OBJECT', 'FIRST CRASH TYPE_HEAD ON',
       'FIRST CRASH TYPE_OTHER NONCOLLISION', 'FIRST CRASH TYPE_OTHER OBJECT',
       'FIRST CRASH TYPE_OVERTURNED', 'FIRST CRASH TYPE_PARKED MOTOR VEHICLE',
       'FIRST CRASH TYPE_PEDALCYCLIST', 'FIRST CRASH TYPE_PEDESTRIAN',
       'FIRST CRASH TYPE_REAR END', 'FIRST CRASH TYPE_REAR TO FRONT',
       'FIRST CRASH TYPE_REAR TO REAR', 'FIRST CRASH TYPE_REAR TO SIDE',
       'FIRST CRASH TYPE_SIDESWIPE OPPOSITE DIRECTION',
       'FIRST CRASH TYPE_SIDESWIPE SAME DIRECTION', 'FIRST CRASH TYPE_TRAIN',
       'FIRST CRASH TYPE_TURNING']
    
    # deal with missing columns
    missing_cols = set(cols_when_model_builds) - set(df_missing.columns)
    for col in missing_cols:
         df_missing[col] = 0
    df = df_missing[cols_when_model_builds]


    from joblib import dump, load
    scaler = load('std_scaler.bin')

    # scaler.fit(preds)
    df = scaler.transform(df)

    # df = pd.DataFrame(df, columns=cols_when_model_builds)
    
    # predict
    result =  model.predict(df)
    fixed_name =  {'FAILING TO YIELD RIGHTOFWAY': 'FAILING TO YIELD RIGHT-OF-WAY',
     'FOLLOWING TOO CLOSELY': 'FOLLOWING TOO CLOSELY',
     'SPEED RELATED': 'SPEED RELATED',
      'IMPROPER BACKING': 'IMPROPER BACKING',
      'IMPROPER OVERTAKINGPASSING': 'IMPROPER OVERTAKING OR PASSING',

      'IMPROPER LANE USAGE': 'IMPROPER LANE USAGE',
      'IMPROPER TURNINGNO SIGNAL': 'IMPROPER TURNINGNO SIGNAL',

      'DISREGARDING TRAFFIC SIGNALS': 'DISREGARDING TRAFFIC SIGNALS',
      'DRIVING SKILLSKNOWLEDGEEXPERIENCE': 'DRIVING SKILLS, KNOWLEDGE OR EXPERIENCE',
      'INTOXICATED/PHYSICAL CONDITION': 'INTOXICATED OR CURRENT PHYSICAL CONDITION RELATED',
      'WEATHER': 'WEATHER',
        'DISTRACTION  FROM OUTSIDE VEHICLE': 'DISTRACTION FROM OUTSIDE THE VEHICLE',
      'DISTRACTION  FROM INSIDE VEHICLE': 'DISTRACTION FROM INSIDE THE VEHICLE',
    'ROAD CONDITIONS': 'ROAD CONDITIONS',
      'EQUIPMENT  VEHICLE CONDITION': 'EQUIPMENT VEHICLE CONDITION'}

    for name, corr_name in fixed_name.items():
        if name in result:
            result = corr_name

    
    # for x in result:
    #     max = sorted(x)
    #      # x.argsort()[0]
    #     if x[max]>0.6:
    #         predictions.append(max)
    #     elif max == 0:
    #         predictions.append(max)
    #     else:
    #         predictions.append(max-1)

    return result


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

        serInput = [speed,age,vision,action,physical,roadway,device,fcrash]

        model = pickle.load(open(filename, 'rb'))

        result =  analysis(speed, age, vision, action, physical, roadway, device, fcrash)
           
        # data=pd.get_dummies(data, columns= ['DRIVER VISION', 'DRIVER ACTION','PHYSICAL CONDITION',
        #        'ROADWAY CONDITION','DEVICE CONDITION', 'FIRST CRASH TYPE'], drop_first=True)

        
        # input = pd.get_dummies(x, columns = CT, drop_first = True)
        # #preds = pd.get_dummies(data, columns = CT, drop_first = True)
        
        # fname = 'scaler.sav'
        # scaler = pickle.load(open(fname, 'rb'))
        

        # preds = scaler.fit_transform(data)

        # preds = np.array(preds)
        # preds = preds.reshape(1,-1)
   
        # preds = model.predict(preds)

        # preds = model_predict(input, model)
        # preds = str(round(preds[0],2))

       
#########################workimg but ignore 
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
    

        return render_template('results.html', predictions=result)
    return None



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(host='0.0.0.0', debug=True) # running the app
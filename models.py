import random 
import numpy as np
import pandas as pd 
import keras
import tensorflow as tf
from datetime import datetime, timedelta


def model_random(df, t_pred):
    # free feel to use data earlier than df.timestamp.min()
    # however, your model SHOULD NOT USE ANY DATA AFTER "t_pred"
    # if hatch "open" should return integer 1
    # if hacth "close" should return integer 0    
    prob_open = random.random()

    return [random.choice([0, 1]), prob_open]

def model_ground_truth(df, t_pred):

    # manually created groud truth for time series, you may create your own groud truth label for a different time series
    # the model should classify hatch as "open" a couple of hours after the pressure drops, return integer 1
    # the model should classify hacth as "close" a couple of hours after the pressure rises, return integer 0    

    t_open = pd.to_datetime('2022-07-16 18:00:00')
    t_clos = pd.to_datetime('2022-08-26 00:00:00')

    if t_pred <= t_open or t_pred >= t_clos:
        status = 0
    else:
        status = 1
    
    
    return status


def search_for_open_hatch_random(df, facility_id):

    assert 'timestamp' in df
    

    num_of_open_hatch_events = random.choice([0, 1, 2]) # replace random choice with your model

    events = []
    if num_of_open_hatch_events>0:
        for i in range(num_of_open_hatch_events):
            open_hatch_event_seq = i + 1
            t_choices = random.choices(df.timestamp.unique(), k=2)

            # because it was random time from the random model, the hatch open time needs to happen between hatch close time, assuming the time series captured hatch open and close events
            t_hacth_open = min(t_choices) 
            t_hacth_clos = max(t_choices)
            prob_has_open_hatch_event = random.random()
            events.append([facility_id, num_of_open_hatch_events, open_hatch_event_seq, t_hacth_open, t_hacth_clos, prob_has_open_hatch_event])

    return events



def model_predict_open_hatch(df:pd.core.frame.DataFrame, t_pred:datetime, model=None):
    #get overall info
    mean = df["pressure_osi"].mean()
    std = df["pressure_osi"].std()
    #get local info
    dfLocal = df[(df.timestamp > t_pred-timedelta(days=2))&
                            (df.timestamp < t_pred)]
    localMean = dfLocal["pressure_osi"].mean()
    localSTD = dfLocal["pressure_osi"].std()
    #build test input
    testInput = tf.constant([[mean, std, localMean, localSTD]])

    #load model from 'models/model' if not already loaded
    if model is None: model = keras.models.load_model("models/model", custom_objects={"test":(lambda x, y: 0)})

    #predict
    prediction = model.predict(testInput, verbose=0)[0][0]
    #fix data that is NaN
    if np.isnan(prediction):
        prediction = 0.499999

    return [int(prediction+0.5), 2*abs(0.5-prediction)]


def search_for_open_hatch(df:pd.core.frame.DataFrame, fac_id:int):

    #we know this is bad, but we are pressed for time at the moment
    pd.set_option('mode.chained_assignment', None)

    #search every 2 days
    search_freq = timedelta(days=2)

    #automatically set start and stop time
    t_strt = pd.to_datetime(df.timestamp.iloc[0])
    t_strt = t_strt.round("D")
    t_strt += timedelta(days=2)
    t_stop = pd.to_datetime(df.timestamp.iloc[-1])
    t_stop = t_stop.round("D")

    df.timestamp = pd.to_datetime(df.timestamp)
    df = df[df.timestamp.between(t_strt, t_stop)]

    #load model from 'models/model' (note: this is for predicting only)
    model = keras.models.load_model("models/model", custom_objects={"error":(lambda x, y: 0)})

    hatch_open = False
    t_open = None
    times = []
    while t_strt <= t_stop:
        
        #add a new datapoint with prediction
        prediction, confidence = model_predict_open_hatch(df, t_strt, model)

        #start a new event:
        if prediction and confidence > 0.9999:
            if not hatch_open:
                hatch_open = True
                t_open = t_strt
        #end an event
        elif hatch_open:
            #only record events that last at least 4 days
            if (t_strt - t_open) > timedelta(days=4):
                times.append((t_open, t_strt))
            hatch_open = False
            
        t_strt += search_freq
    
    if hatch_open:
        times.append((t_open, t_stop))
    
    events = []
    events = [(fac_id, len(times), i, opened, closed) for i, (opened, closed) in enumerate(times)]
    
    return events
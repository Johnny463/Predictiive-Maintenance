import joblib
from fastapi import FastAPI
from pydantic import BaseModel

class SensorData(BaseModel):
    machineID: int = 1
    voltmean_3h: float = 178.59911946468
    temperaturemean_3h: float = 440.93120256800734
    pressuremean_3h: float = 94.79592637844898
    vibrationmean_3h: float = 44.673604385155436
    voltsd_3h: float = 12.742676402748366
    temperaturesd_3h: float = 11.85203821080354
    pressuresd_3h: float = 23.934573069226253
    vibrationsd_3h: float = 2.913937327991136
    voltmean_24h: float = 175.48305457957895
    temperaturemean_24h: float = 461.06516471717896
    pressuremean_24h: float = 98.47689682867882
    vibrationmean_24h: float = 42.75041814287485
    voltsd_24h: float = 13.689807902230038
    temperaturesd_24h: float = 41.4928882421844
    pressuresd_24h: float = 9.801411865756965
    vibrationsd_24h: float = 5.982448529515937
    error1count: float = 0.0
    error2count: float = 0.0
    error3count: float = 0.0
    error4count: float = 0.0
    error5count: float = 1.0
    comp1: float = 29.125
    comp2: float = 59.125
    comp3: float = 89.125
    comp4: float = 74.125
    age: int = 18
    model_model1: bool = False
    model_model2: bool = False
    model_model3: bool = True
    model_model4: bool = False

app = FastAPI()

# Load your model
model = joblib.load("model.pkl")  # Replace "model_filename.pkl" with the filename of your saved model

@app.post("/predict/")
async def predict(sensor_data: SensorData):
    # Convert the input data to a format that can be used for prediction
    input_data = [[
        sensor_data.machineID,
        sensor_data.voltmean_3h,
        sensor_data.temperaturemean_3h,
        sensor_data.pressuremean_3h,
        sensor_data.vibrationmean_3h,
        sensor_data.voltsd_3h,
        sensor_data.temperaturesd_3h,
        sensor_data.pressuresd_3h,
        sensor_data.vibrationsd_3h,
        sensor_data.voltmean_24h,
        sensor_data.temperaturemean_24h,
        sensor_data.pressuremean_24h,
        sensor_data.vibrationmean_24h,
        sensor_data.voltsd_24h,
        sensor_data.temperaturesd_24h,
        sensor_data.pressuresd_24h,
        sensor_data.vibrationsd_24h,
        sensor_data.error1count,
        sensor_data.error2count,
        sensor_data.error3count,
        sensor_data.error4count,
        sensor_data.error5count,
        sensor_data.comp1,
        sensor_data.comp2,
        sensor_data.comp3,
        sensor_data.comp4,
        sensor_data.age,
        sensor_data.model_model1,
        sensor_data.model_model2,
        sensor_data.model_model3,
        sensor_data.model_model4
    ]]
    
    # Perform prediction
    proba = model.predict_proba(input_data)
    print(proba)
    
    # Convert the probabilities to a list
    probabilities = proba.tolist()[0]
    
    return {"probabilities": probabilities[0]}

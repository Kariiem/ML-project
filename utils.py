import pickle

Alcohol_str_to_int = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}
Food_between_Meals_str_to_int = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}
Transport_str_to_int = {
    "Public_Transportation": 0,
    "Automobile": 1,
    "Motorbike": 2,
    "Bike": 3,
    "Walking": 4,
}
Body_Level_str_to_int = {
    "Body Level 1": 0,
    "Body Level 2": 1,
    "Body Level 3": 2,
    "Body Level 4": 3,
}
transform_dataset = {
    "Gender": lambda x: int(x == "Male"),
    "Age": lambda x: round(x),
    "Height": lambda x: x,
    "Weight": lambda x: x,
    "H_Cal_Consump": lambda x: int(x == "yes"),
    "Veg_Consump": lambda x: x,
    "Water_Consump": lambda x: x,
    "Alcohol_Consump": lambda x: Alcohol_str_to_int[x],
    "Smoking": lambda x: int(x == "yes"),
    "Meal_Count": lambda x: x,
    "Food_Between_Meals": lambda x: Food_between_Meals_str_to_int[x],
    "Fam_Hist": lambda x: int(x == "yes"),
    "H_Cal_Burn": lambda x: int(x == "yes"),
    "Phys_Act": lambda x: x,
    "Time_E_Dev": lambda x: x,
    "Transport": lambda x: Transport_str_to_int[x],
    "Body_Level": lambda x: Body_Level_str_to_int[x],
}


def save_model(model):
    model_name = model.__class__.__name__
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    model = None
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

transform_dataset_round = {
    "Gender": lambda x: int(x == "Male"),
    "Age": lambda x: round(x),
    "Height": lambda x: x,
    "Weight": lambda x: x,
    "H_Cal_Consump": lambda x: int(x == "yes"),
    "Veg_Consump": lambda x: round(x),
    "Water_Consump": lambda x: round(x),
    "Alcohol_Consump": lambda x: Alcohol_str_to_int[x],
    "Smoking": lambda x: int(x == "yes"),
    "Meal_Count": lambda x: round(x),
    "Food_Between_Meals": lambda x: Food_between_Meals_str_to_int[x],
    "Fam_Hist": lambda x: int(x == "yes"),
    "H_Cal_Burn": lambda x: int(x == "yes"),
    "Phys_Act": lambda x: round(x),
    "Time_E_Dev": lambda x: round(x),
    "Transport": lambda x: Transport_str_to_int[x],
    "Body_Level": lambda x: Body_Level_str_to_int[x],
}
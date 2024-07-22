from obligation_activity_classifier import classify_obligation_activities
from obligation_classifier import classify_obligations
from obligation_type_classifier import classify_obligation_type


def prediction(input_text: str):
    obligation_prediction = classify_obligations(input_text)

    # only proceed with other predictions if the obligation is classified as an obligation
    # otherwise, return the input data as is
    if obligation_prediction == "Obligation":
        obligation_activities = classify_obligation_activities(input_text)
        obligation_type = classify_obligation_type(input_text)

        return {
            "text": input_text,
            "prediction": {
                "is_obligation_choice": obligation_prediction,
                "obligation_category_choice": obligation_type,
                "obligation_activity_choices": obligation_activities
            }
        }
    else:
        return {
            "text": input_text,
            "prediction": {
                "is_obligation_choice": obligation_prediction
            }
        }

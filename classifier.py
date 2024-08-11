from activity_classifier import classify_activities
from obligation_classifier import classify_obligations
from type_classifier import classify_type


def prediction(input_text: str):
    obligation_prediction = classify_obligations(input_text)

    # only proceed with other predictions if the obligation is classified as an obligation
    # otherwise, return the input data as is
    if obligation_prediction == "Obligation":
        return {
            "text": input_text,
            "prediction": {
                "is_obligation_choice": obligation_prediction,
                "obligation_category_choice": classify_type(input_text),
                "obligation_activity_choices": classify_activities(input_text)
            }
        }
    else:
        return {
            "text": input_text,
            "prediction": {
                "is_obligation_choice": obligation_prediction
            }
        }

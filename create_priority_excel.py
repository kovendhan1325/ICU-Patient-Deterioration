import pandas as pd
import os

def create_priority_excel(excel_path):
    data = [
        {"Rank": 1, "Feature": "shock_index", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Highly sensitive early indicator of shock or internal bleeding."},
        {"Rank": 2, "Feature": "respiration", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Often the very first vital sign to spike before respiratory failure or arrest."},
        {"Rank": 3, "Feature": "systemicsystolic", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Dropping systolic pressure indicates loss of organ perfusion."},
        {"Rank": 4, "Feature": "systemicdiastolic", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Component of blood pressure, crucial for heart perfusion."},
        {"Rank": 5, "Feature": "systemicmean", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Mean arterial pressure is the most important metric for brain/kidney blood flow."},
        {"Rank": 6, "Feature": "heartrate", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Spikes during distress/infection, drops right before cardiac arrest."},
        {"Rank": 7, "Feature": "map_calculated", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Derived MAP metric, confirms organ perfusion status."},
        {"Rank": 8, "Feature": "pulse_pressure", "Priority Tier": "Tier 1: Critical Real-Time Indicators", "Clinical Reasoning": "Narrows significantly during heart failure or severe blood loss."},
        {"Rank": 9, "Feature": "Lactate", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "The ultimate marker for sepsis and tissue starvation. High = cells dying."},
        {"Rank": 10, "Feature": "Creatinine", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "Indicates acute kidney failure."},
        {"Rank": 11, "Feature": "Bilirubin", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "Indicates liver failure."},
        {"Rank": 12, "Feature": "WBC", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "Indicates massive infection or severe immune response (e.g. sepsis)."},
        {"Rank": 13, "Feature": "Platelets", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "Low platelets indicate severe bleeding risks or late-stage sepsis (DIC)."},
        {"Rank": 14, "Feature": "temperature", "Priority Tier": "Tier 2: Organ Damage & Infection Markers", "Clinical Reasoning": "Severe fever (sepsis) or severe hypothermia (shock)."},
        {"Rank": 15, "Feature": "heartrate_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures if the heart rate is actively crashing or spiking right now."},
        {"Rank": 16, "Feature": "systemicsystolic_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures active blood pressure drops."},
        {"Rank": 17, "Feature": "systemicdiastolic_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures active blood pressure drops."},
        {"Rank": 18, "Feature": "systemicmean_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures active MAP drops."},
        {"Rank": 19, "Feature": "respiration_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures sudden onset of respiratory distress."},
        {"Rank": 20, "Feature": "temperature_rate_of_change", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Captures sudden fever spikes or drops."},
        {"Rank": 21, "Feature": "heartrate_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "High variability indicates the body is struggling to maintain stability."},
        {"Rank": 22, "Feature": "systemicsystolic_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "High variability indicates cardiovascular instability."},
        {"Rank": 23, "Feature": "systemicdiastolic_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "High variability indicates cardiovascular instability."},
        {"Rank": 24, "Feature": "systemicmean_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "High variability indicates cardiovascular instability."},
        {"Rank": 25, "Feature": "respiration_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "High variability indicates respiratory exhaustion."},
        {"Rank": 26, "Feature": "temperature_rolling_std", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "Temperature volatility."},
        {"Rank": 27, "Feature": "heartrate_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average heart rate over 6 hours."},
        {"Rank": 28, "Feature": "systemicsystolic_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average systolic BP over 6 hours."},
        {"Rank": 29, "Feature": "systemicdiastolic_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average diastolic BP over 6 hours."},
        {"Rank": 30, "Feature": "systemicmean_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average MAP over 6 hours."},
        {"Rank": 31, "Feature": "respiration_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average respiration rate over 6 hours."},
        {"Rank": 32, "Feature": "temperature_rolling_mean", "Priority Tier": "Tier 3: The 6-Hour Trajectories", "Clinical Reasoning": "The sustained average temperature over 6 hours."},
        {"Rank": 33, "Feature": "resp_fio2", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Oxygen concentration given by ventilator (spikes if lungs failing)."},
        {"Rank": 34, "Feature": "resp_fio2_(%)", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Alternative oxygen concentration given by ventilator."},
        {"Rank": 35, "Feature": "resp_peep", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Ventilator pressure required to keep lungs open."},
        {"Rank": 36, "Feature": "Hemoglobin", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Indicates slow blood loss or anemia."},
        {"Rank": 37, "Feature": "Potassium", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Electrolyte imbalance (can cause arrhythmias)."},
        {"Rank": 38, "Feature": "Sodium", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Electrolyte imbalance (dehydration, kidney issues)."},
        {"Rank": 39, "Feature": "Glucose", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Blood sugar levels (erratic in ICU but rarely the primary cause of sudden death)."},
        {"Rank": 40, "Feature": "resp_peep/cpap", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Additional respiratory support metrics."},
        {"Rank": 41, "Feature": "resp_ps_above_peep", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Additional respiratory support metrics."},
        {"Rank": 42, "Feature": "resp_set_fraction_of_inspired_oxygen_(fio2)", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Additional respiratory support metrics."},
        {"Rank": 43, "Feature": "resp_unable_to_obtain_peepi_and_vtrap", "Priority Tier": "Tier 4: Routine Labs & Resp Support", "Clinical Reasoning": "Machine read failure flag/metric."},
        {"Rank": 44, "Feature": "age", "Priority Tier": "Tier 5: Static Measurements", "Clinical Reasoning": "Older patients have less physiological reserve to survive a shock."},
        {"Rank": 45, "Feature": "bmi", "Priority Tier": "Tier 5: Static Measurements", "Clinical Reasoning": "Extreme obesity or malnourishment adds risk factor."},
        {"Rank": 46, "Feature": "admissionweight", "Priority Tier": "Tier 5: Static Measurements", "Clinical Reasoning": "Baseline static metric."},
        {"Rank": 47, "Feature": "admissionheight", "Priority Tier": "Tier 5: Static Measurements", "Clinical Reasoning": "Baseline static metric."}
    ]
    
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)
    print(f"Successfully saved priority rankings to {excel_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, "continuous_features_priority.xlsx")
    create_priority_excel(excel_path)

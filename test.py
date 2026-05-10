import pandas as pd

print("Loading big PISA file...")

df = pd.read_spss("data/CY08MSP_STU_QQQ.sav")

columns = [
    "ESCS",
    "HOMEPOS",
    "ICTRES",
    "ICTHOME",
    "ST004D01T",
    "ST005Q01JA",
    "ST006Q01JA",
    "ST006Q02JA",
    "ST006Q03JA",
    "ST007Q01JA",
    "WORKPAY",
    "WORKHOME",
    "MATHPREF",
    "MATHEASE",
    "MATHMOT",
    "BELONG",
    "BULLIED",
    "FAMSUP",
    "TEACHSUP",
    "MATHEFF",
    "FAMCON",
    "ANXMAT",
    "MATHPERS",
    "W_FSTUWT",
    "PV1MATH",
    "PV2MATH",
    "PV3MATH",
    "PV4MATH",
    "PV5MATH"
]

df = df[columns]

df = df.sample(100000, random_state=42)

df.to_csv("data/sample.csv", index=False)

print("Saved data/sample.csv")
print("Shape:", df.shape)


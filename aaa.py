import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    hnr05 = call(harmonicity05, "Get mean", 0, 0)
    harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
    hnr15 = call(harmonicity15, "Get mean", 0, 0)
    harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    hnr25 = call(harmonicity25, "Get mean", 0, 0)
    harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
    hnr35 = call(harmonicity35, "Get mean", 0, 0)
    harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
    hnr38 = call(harmonicity38, "Get mean", 0, 0)
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38

localJitter_list = [] #measure
localabsoluteJitter_list = [] #measure
rapJitter_list = [] #measure
ppq5Jitter_list = [] #measure
localShimmer_list =  [] #measure
localdbShimmer_list = [] #measure
apq3Shimmer_list = [] #measure
aqpq5Shimmer_list = [] #measure
apq11Shimmer_list =  [] #measure
hnr05_list = [] #measure
hnr15_list = [] #measure
hnr25_list = [] #measure
parkinson_list = [] #Parkinson(1) or healthy(0)
file_list = []

for wave_file in glob.glob("audio/SpontaneousDialogue/PD/*.wav"):
    sound = parselmouth.Sound(wave_file)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    file_list.append(wave_file) # make an ID list
    localJitter_list.append(localJitter) # make a mean F0 list
    localabsoluteJitter_list.append(localabsoluteJitter) # make a sd F0 list
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    hnr05_list.append(hnr05)
    hnr15_list.append(hnr15)
    hnr25_list.append(hnr25)
    parkinson_list.append(1) #1 because parkinson file
    
for wave_file in glob.glob("audio/ReadText/PD/*.wav"):
    sound = parselmouth.Sound(wave_file)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    file_list.append(wave_file) # make an ID list
    localJitter_list.append(localJitter) # make a mean F0 list
    localabsoluteJitter_list.append(localabsoluteJitter) # make a sd F0 list
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    hnr05_list.append(hnr05)
    hnr15_list.append(hnr15)
    hnr25_list.append(hnr25)
    parkinson_list.append(1) #1 because parkinson file
    
for wave_file in glob.glob("audio/SpontaneousDialogue/HC/*.wav"):
    sound = parselmouth.Sound(wave_file)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    file_list.append(wave_file) # make an ID list
    localJitter_list.append(localJitter) # make a mean F0 list
    localabsoluteJitter_list.append(localabsoluteJitter) # make a sd F0 list
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    hnr05_list.append(hnr05)
    hnr15_list.append(hnr15)
    hnr25_list.append(hnr25)
    parkinson_list.append(0) #0 because healthy file
    
for wave_file in glob.glob("audio/ReadText/HC/*.wav"):
    sound = parselmouth.Sound(wave_file)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    file_list.append(wave_file) # make an ID list
    localJitter_list.append(localJitter) # make a mean F0 list
    localabsoluteJitter_list.append(localabsoluteJitter) # make a sd F0 list
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    hnr05_list.append(hnr05)
    hnr15_list.append(hnr15)
    hnr25_list.append(hnr25)
    parkinson_list.append(0) #0 because healthy file



pred = pd.DataFrame(np.column_stack([parkinson_list,localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, hnr05_list, hnr15_list, hnr25_list]),
                               columns=["Parkinson","Jitter_rel","Jitter_abs","Jitter_RAP","Jitter_PPQ","Shim_loc","Shim_dB","Shim_APQ3","Shim_APQ5","Shi_APQ11", "hnr05", "hnr15", "hnr25"])  #add these lists to pandas in the right order

pred['hnr25'].fillna((pred['hnr25'].mean()), inplace=True) #Data cleaning because they may be NaN values
pred['hnr15'].fillna((pred['hnr15'].mean()), inplace=True) #Data cleaning because they may be NaN values

pred.to_csv("processed_results.csv", index=False) # Write out the updated dataset

parkinson = pd.read_csv("processed_results.csv") #Loading CSV dataset

predictors=["Jitter_rel","Jitter_abs","Jitter_RAP","Jitter_PPQ","Shim_loc","Shim_dB","Shim_APQ3","Shim_APQ5","Shi_APQ11","hnr05","hnr15", "hnr25"] #Listing predictors

for col in predictors: # Loop through all columns in predictors
    if parkinson[col].dtype == 'object':  # check if column's type is object (text)
        parkinson[col] = pd.Categorical(parkinson[col]).codes  # convert text to numerical

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(parkinson[predictors], parkinson['Parkinson'], test_size=0.25, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVC

#1. LogisticRegression模型
clf = LogisticRegression()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print ('train accuracy =', train_score)
print ('test accuracy =', test_score)

#train accuracy = 0.6666666666666666
#test accuracy = 0.631578947368421


#2. LASSO模型
clf2 = Lasso(alpha=0.01)
clf2.fit(X_train, y_train)
train_score2 = clf2.score(X_train, y_train)
test_score2 = clf2.score(X_test, y_test)

print ('train accuracy =', train_score2)
print ('test accuracy =', test_score2)
#train accuracy = 0.201490197819296
#test accuracy = 0.15157538210189192

#3. SVM rbf内核模型
clf3 = SVC(kernel='rbf', C=1)#创建SVM训练模型
clf3.fit(X_train,y_train)#对训练集数据进行训练
train_score3 = clf3.score(X_train, y_train)
test_score3 = clf3.score(X_test, y_test)

print ('train accuracy =', train_score3)
print ('test accuracy =', test_score3)
#train accuracy = 0.5925925925925926
#test accuracy = 0.5789473684210527


#4. SVM linear内核模型
clf4 = SVC(kernel='linear', C=1, probability = True)#创建SVM训练模型
clf4.fit(X_train,y_train)#对训练集数据进行训练
train_score4 = clf4.score(X_train, y_train)
test_score4 = clf4.score(X_test, y_test)

print ('train accuracy =', train_score4)
print ('test accuracy =', test_score4)
#train accuracy = 0.6481481481481481
#test accuracy = 0.631578947368421



import joblib

clf.fit(X_train, y_train)
joblib.dump(clf, "trainedModel.sav")

clf.fit(X_train, y_train)
joblib.dump(clf2, "trainedModel2.sav")

clf3.fit(X_train, y_train)
joblib.dump(clf3, "trainedModel3.sav")

clf4.fit(X_train, y_train)
joblib.dump(clf4, "trainedModel4.sav")
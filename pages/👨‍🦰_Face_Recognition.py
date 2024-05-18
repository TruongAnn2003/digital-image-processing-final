import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import tempfile


st.subheader('Nhận dạng khuôn mặt')
FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

press = st.button('Stop')
if press:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('face_recognition/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

svc = joblib.load('face_recognition/svc.pkl')
mydict = ['An', 'AnhKhoa', 'Cong', 'MinhDuc', 'ThanhLoi']

def visualize(results, input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.putText(input, results[idx],(coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        'face_recognition/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    'face_recognition/face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    video_file = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'mkv'])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv.VideoCapture(tfile.name)
    else:
        cap = cv.VideoCapture(0)

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        faces = detector.detect(frame) 
        tm.stop()
        results = []
        if faces[1] is not None:
            for face in faces[1]:
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                results.append(result)
        visualize(results, frame, faces, tm.getFPS())

        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()

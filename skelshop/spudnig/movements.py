def rest(i, data, keypoint):
    """Determines whether current point is in rest position"""
    x = data.loc[i][keypoint]
    y = data.loc[i][keypoint + 1]
    span = 14

    start = int(i - span / 2)
    if start < 0:
        start = 0
    end = int(i + span / 2)
    if end > data.shape[0]:
        end = data.shape[0]

    certainty = 0
    for j in range(start, end):
        if (
            abs(data.loc[j][keypoint] - x) < 10
            and abs(data.loc[j][keypoint + 1] - y) < 10
        ):
            certainty += 1

    if certainty / span >= 0.7:
        return True
    else:
        return False


def isStill(data, idx, keypoint):
    """Returns true if the current idx point is resting"""
    count = 0
    current = data.loc[idx]
    for x in range(min(idx + 1, data.shape[0]), min(idx + 21, data.shape[0])):
        if (
            abs(current[keypoint] - data.loc[x][keypoint]) < 8
            and abs(current[keypoint + 1] - data.loc[x][keypoint + 1]) < 8
        ):
            count += 1
    if count / 20 > 0.7:
        return True
    else:
        return False


def get_gestures(data, keypoint, threshold=0.3):
    """Recognizes gestures that occur in data based on keypoint 'keypoint' and stores outcomes in gestures."""
    local_gestures = [0]
    rest_x = 0
    rest_y = 0
    keypoint = keypoint * 3

    i = 0
    # fill right hand gestures
    while i < data.shape[0] - 1:
        i = i + 1
        gesture = 0
        current = data.loc[i]
        # prev = data.loc[i - 1]
        returned = False
        backToRest = -1

        # if certainty (of openpose) is too low there is no gesture
        if current[keypoint + 2] < threshold:
            local_gestures.append(gesture)

        else:
            # update rest position
            if rest(i, data, keypoint):
                rest_x = current[keypoint]
                rest_y = current[keypoint + 1]
                gesture = 0

            # if hand coordinates are different from previous frame
            elif (
                abs(current[keypoint] - rest_x) > 5
                or abs(current[keypoint + 1] - rest_y) > 5
            ):
                # check if it's actual movement or just a few frames
                certainty = 0
                for x in range(i + 1, min(i + 6, data.shape[0])):
                    if (
                        abs(data.loc[x][keypoint] - rest_x) > 5
                        or abs(data.loc[x][keypoint + 1] - rest_y) > 5
                    ):
                        certainty += 1
                # if there is no movement
                if certainty / 5 < 0.5:
                    gesture = 0
                # if there is indeed movement
                else:
                    # TODO: check whether hands return to rest position in future: if so, there is a gesture from current i until it reaches rest point again
                    for t in range(i + 1, min(i + 300, data.shape[0])):
                        # if hand has returned to rest position in next 10 seconds
                        if isStill(data, t, keypoint):
                            gesture = 1
                            returned = True
                            backToRest = t
                            rest_x = data.loc[t][keypoint]
                            rest_y = data.loc[t][keypoint + 1]
                            break

                    # if hands did not return to previous rest position, check if they returned to new restposition
                    if not returned:
                        for t in range(i + 1, min(i + 300, data.shape[0])):
                            if rest(t, data, keypoint):
                                gesture = 1
                                returned = True
                                backToRest = t
                                rest_x = data.loc[backToRest][keypoint]
                                rest_y = data.loc[backToRest][keypoint + 1]
                                break

            if returned:
                for k in range(i, backToRest + 1):
                    local_gestures.append(gesture)
                i = backToRest
            else:
                local_gestures.append(gesture)

    return local_gestures

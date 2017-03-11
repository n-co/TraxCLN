import logging


def stop_and_read(run_mode):
    if run_mode == 'debug':
        raw_input("Type anything to continue. ")


def learn_about(context, run_mode):
    ans = True
    subject = context[0][11][0]
    logging.warn("recieved context param. this is a python list! it should be of the same "
                 "size as hidden layers. length: " + str(len(context)))
    logging.warn("we will analyze the first layer.")
    logging.warn("the shape of context[0], ie the shape of context for first layer: " + str(context[0].shape))
    logging.warn("selecting the 12th example, and the first relation from this, we have an"
                 "array of shape: " + str(subject.shape))
    logging.warn("the contents of this array are: " + str(subject))
    stop_and_read(run_mode)
    return ans

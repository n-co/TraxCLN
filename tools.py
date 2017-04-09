import logging


def stop_and_read(run_mode):
    if run_mode == 'debug':
        raw_input("Type anything to continue. ")


def learn_about(context, run_mode):
    ans = True
    subject = context[0][1][0]
    # logging.debug("recieved context param. this is a python list! it should be of the same "
    #              "size as hidden layers. length: " + str(len(context)))
    # logging.debug("we will analyze the first layer.")
    # logging.debug("the shape of context[0], ie the shape of context for first layer: " + str(context[0].shape))
    # logging.debug("selecting the 12th example, and the first relation from this, we have an"
    #              "array of shape: " + str(subject.shape))
    # logging.debug("the contents of this array are: " + str(subject))
    stop_and_read(run_mode)
    return ans


def expected_run_time(epochs, batch_size, sample_size, batch_rate):
    ans = sample_size / batch_size
    ans *= batch_rate
    ans *= epochs
    ans = ans / 60.0 / 60.0 / 24.0
    return ans


def required_batch_rate(epochs, batch_size, sample_size, run_time):
    ans = run_time * batch_size
    ans = ans / epochs
    ans = ans * 60.0 * 60.0 * 24.0
    ans = ans / sample_size
    return ans


print expected_run_time(10,256,60000,25)
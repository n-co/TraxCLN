from config import *
import operator

filter_limit = 1000
filtered_csv_path = raw_data_dir + '/data_filtered.csv'


def count_labels(csv_path):
    """
    according to a trax formatted csv file, count instances of each label
    :param csv_path: path to a csv file.
    :return: a dictionary with key=label, value=label_count
    """
    logging.info("labels_counter: Started.")
    labels_dict = {}
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        i = 0
        for row in reader:
            label = row[10]
            if label in labels_dict:
                labels_dict[label] += 1
            else:
                labels_dict[label] = 1

            if i % 50000 == 0:
                logging.debug("index now at: %d" % i)
            i += 1
    logging.info("labels_counter: Ended.")
    return labels_dict


def print_dict(dictionary):
    logging.info("print_dict: Started.")
    # return a list of tuples, when each tuple is (key,value), that ordered in the list by value
    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    for tup in sorted_dict:
        print '%40s: %s' % (tup[0], tup[1])
    logging.info("print_dict: Ended.")


def count_above_limit(dictionary, limit):
    """
    a helper function, that counts how many keys (labels) there are with with value >= limit, in the dictionary,
    and sums also all theses values
    i.e. how many labels appears more than "limit" times, and how many products holds those labels
    :param dictionary: (key=label, value=count_in_csv)
    :param limit:
    :return: a tuple (how_many_labels, how_many_products)
    """
    prod_counter = 0
    labels_counter = 0
    for key in dictionary:
        if dictionary[key] >= limit:
            prod_counter += dictionary[key]
            labels_counter += 1
    return labels_counter, prod_counter


def create_filtered_csv(original_csv_path, new_csv_path, limit_filter, labels_dictionary):
    """
    takes the original csv file and filters out every row (product) that it's label isn't common
    (i.e. appears less than 'limit_filter' times in the original csv file)
    :param original_csv_path:
    :param new_csv_path:
    :param limit_filter:
    :param labels_dictionary:
    :return: nothing
    """
    logging.info("create_filtered_csv: Started.")

    with open(original_csv_path, 'r') as original:
        lines = itertools.islice(original, 0, None)
        reader = csv.reader(lines)
        field_names = reader.next()

        with open(new_csv_path, 'w') as filtered:
            writer = csv.writer(filtered)
            writer.writerow(field_names)
            i = 0
            row_id = 0
            for row in reader:
                label = row[10]
                if labels_dictionary[label] >= limit_filter:
                    row[0] = str(row_id)
                    writer.writerow(row)
                    row_id += 1
                if i % 50000 == 0:
                    logging.debug("index now at: %d" % i)
                i += 1

    logging.info("create_filtered_csv: Ended.")

labels_dict = count_labels(csv_path)
print_dict(labels_dict)
# limits = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# for limit in limits:
#     labels, products = count_above_limit(labels_dict, limit)
#     print 'limit = %4d   relevant labels = %4d   relevant products = %6d' % (limit, labels, products)

create_filtered_csv(csv_path, filtered_csv_path, filter_limit, labels_dict)

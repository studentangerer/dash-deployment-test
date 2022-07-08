import simpy
import random
import csv
import json
from scipy.stats import poisson
import numpy as np
from collections import defaultdict
import logging

from collections import defaultdict

import dash  # version 1.13.1
from dash import dcc, html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np

import dash  # version 1.13.1
from dash import dcc, html
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np
import data_visualization




logging.basicConfig(level=logging.INFO)


# g class contains global parameter values
class g:
    num_lambda = [4, 15, 65]
    prob_card = [0.5, 0.6, 0.95]
    onequeue = True
    with open('ItemScanTimes_Supermarkt.csv', 'r') as f:
        file = csv.reader(f)
        item_scan_times = list(file)

    sim_data = []

class data_storage:
    plott_data = []

# Class representing customers
class Customer:
    def __init__(self, c_id, num_lambda, prob_card, counter, id_counter, env, dc):
        self.id = c_id
        self.num_arcticles = poisson.rvs(num_lambda)
        self.prob_card = prob_card
        self.counter = counter
        self.id_counter = id_counter
        self.env = env
        self.data_collector = dc
        self.card_payment = self.determine_if_card_payment()
    # this methods calculates the time a customer need to pay. with prob_card, cus. pays by card. with a prob. of 1%
    # this transaction fails. the cus. repeats it up to 4 times and then continues to pay in cash
    def get_payment_times(self):
        time_to_pay = 0
        # if the customer pays by card
        if self.card_payment:
            num_card_tries = 0
            # the payment process by card can fail up to 3 times
            while num_card_tries < 3:
                # if the payment process by card fails
                if random.random() < 0.01:
                    # it counts the amout of faild tries and also adds up the time it takes for each card payment
                    num_card_tries += 1
                    time_to_pay += random.randint(20, 45)
                # in the 99% of cases the card process goes through normaly
                else:
                    time_to_pay = random.randint(20, 45)
                    return time_to_pay
            # this is the case if the card payment failed up to 3 times. then the cus. pays by cash
            time_to_pay += random.randint(20, 60)
        # this is the regular case where the customer wanted to pay cash from the beginning
        else:
            time_to_pay = random.randint(20, 60)
        return time_to_pay

    def get_scan_times(self):
        time_to_scan = 0
        for _ in range(1, self.num_arcticles):
            # in 1 % of cases the scanner fails. then manual action is required which takes between 10 and 25 sec
            if random.random() < 0.01:
                time_to_scan += random.randint(10, 25)
            else:
                time_to_scan += float(g.item_scan_times[random.randint(0, 149)][0])
        return time_to_scan

    def determine_if_card_payment(self):
        if random.random() < self.prob_card:
            return True
        else:
            return False

    def start_checkout(self):
        logging.debug(
            f"Customer {self.id} in queue {self.id_counter} with {self.num_arcticles} articles at timepoint: {self.env.now}.")
        # register customer arrival
        self.data_collector.register_arrival(self.id, self.id_counter, self.card_payment, self.num_arcticles)
        arr = self.env.now
        with self.counter.request() as req:
            yield req

            self.data_collector.register_start_scanning(self.id)
            time_to_scan = self.get_scan_times()
            # logging.debug(f"Customer {self.id} started scanning at {round(self.env.now, 2)}")
            # logging.debug(f"Customer {self.id} at line {self.id_counter}")
            yield self.env.timeout(time_to_scan)
            DataCollector.customer_log[self.id].append({'type': 'ended_scanning', 'time': self.env.now})
            # logging.debug(f"Customer {self.id} ended scanning at {round(self.env.now, 2)}")

            time_to_pay = self.get_payment_times()
            DataCollector.customer_log[self.id].append({'type': 'start_paying', 'time': self.env.now})
            # logging.debug(f"Customer {self.id} started paying at {round(self.env.now, 2)}")
            yield self.env.timeout(time_to_pay)
            # logging.debug(f"Customer {self.id} endet paying at {round(self.env.now, 2)}")
            self.data_collector.register_end(self.id, self.id_counter)


# Class representing the model
class CheckoutModel:
    # Constructor sets up SimPy environment, sets customer counter to 0 (used to assign customer id)
    # and sets up our resources ( includes a checkout resource, with a capacity given by the g class
    def __init__(self, sim_duration, onequeue, num_checkout, prob_cust_spon, prob_cust_reg, prob_cust_stock,
                 distribution_type, dist_value, data_visual, queue_desicion):
        self.env = simpy.Environment()
        self.sim_duration = sim_duration
        self.customer_counter = 0
        self.num_queues = 1
        self.weights = [prob_cust_spon, prob_cust_reg, prob_cust_stock]
        self.checkout = []
        self.dcs = []
        self.distribution_type = distribution_type
        self.distribution_value = dist_value
        self.data_visual = data_visual
        self.queue_desicion = queue_desicion

        # Creating checkout-points either with one queue each or together
        if onequeue:
            self.checkout.append(simpy.Resource(self.env, capacity=num_checkout))
            self.dcs.append(DataCollector(self.env))
        else:
            self.num_queues = num_checkout
            for _ in range(num_checkout):
                self.checkout.append(simpy.Resource(self.env, capacity=1))
                self.dcs.append(DataCollector(self.env))

    # A Method that generates customers arriving at the checkout
    def generate_customer_arrivals(self):
        # Keep generating indefinitly (until the simulation ends)
        while self.env.now <= self.sim_duration:
            self.customer_counter += 1
            # Randomly choose a customer type
            kind = random.choices([0, 1, 2], self.weights)[0]
            # Randomly choose a Queue
            match self.queue_desicion:
                case "shortest":
                    id_checkout = DataCollector.get_shortest_queue_number(self.num_queues)
                case "random":
                    id_checkout = random.randint(0, self.num_queues - 1)
                case _:
                    id_checkout = random.randint(0, self.num_queues - 1)
            # Create a new Customer - an instance of the Customer Class
            # and give the customer an ID determined by the customer counter
            cc = Customer(self.customer_counter, g.num_lambda[kind], g.prob_card[kind],
                          self.checkout[id_checkout], id_checkout + 1, self.env, self.dcs[id_checkout])
            # Get the SimPy environment to run the  with this customer
            self.env.process(cc.start_checkout())

            # Randomly sample the time to the next customer arriving for the checkout
            match self.distribution_type:
                case exponential:
                    sampled_arrival_time = np.random.exponential(self.distribution_value)

            # Freeze this function until that time has elapsed.
            yield self.env.timeout(sampled_arrival_time)

    # The run method starts u the entity genrators, and tells SimPy to start running the environment
    def run(self):
        self.env.process(self.generate_customer_arrivals())
        self.env.run()
        DataCollector.merged_results(self.dcs)
        match self.data_visual:
            case 'matplotlib':
                DataCollector.visualize_sim_data(self.dcs)
            case 'dashboard':
                DataCollector.save_simulation_data_global(self.dcs)



class DataCollector:
    customer_log = defaultdict(lambda: [{}])
    customer_position_log = defaultdict(lambda: [{}])
    customer_line_log = defaultdict(lambda: [{'time': 0, 'line_length': 0}])

    def __init__(self, env):
        self.env = env
        self.line_log = [{'time': 0, 'line_length': 0}]

    def register_arrival(self, id_customer, id_counter, card_payment, number_articles):
        old_line_length = self.line_log[len(self.line_log) - 1]["line_length"]
        self.line_log.append({'time': self.env.now, 'line_length': old_line_length + 1})
        DataCollector.customer_line_log[id_counter].append({'time': self.env.now, 'line_length': old_line_length + 1})
        DataCollector.customer_log[id_customer].append(
            {'type': 'created', 'time': self.env.now, 'counter_id': id_counter})
        DataCollector.customer_position_log[id_customer].append({'time': self.env.now,
                                                                 'position': self.calculate_position(id_counter),
                                                                 'id_counter': id_counter,
                                                                 'card_payment': card_payment,
                                                                 'number_articles': number_articles})

    def register_start_scanning(self, id_customer):
        DataCollector.customer_log[id_customer].append({'type': 'start_scanning', 'time': self.env.now})

    def register_end(self, id_customer, id_counter):
        old_line_length = self.line_log[len(self.line_log) - 1]["line_length"]
        self.line_log.append({'time': self.env.now, 'line_length': old_line_length - 1})
        DataCollector.customer_log[id_customer].append({'type': 'ended_paying', 'time': self.env.now})
        DataCollector.customer_line_log[id_counter].append({'time': self.env.now, 'line_length': old_line_length - 1})
        self.update_customer_positions(id_counter)

    def result_line(self):
        avg_queue_length = 0
        for i in range(1, len(self.line_log)):
            avg_queue_length += (self.line_log[i]["time"] - self.line_log[i - 1]["time"]) * self.line_log[i][
                "line_length"]
        avg_queue_length = avg_queue_length / self.env.now
        return avg_queue_length

    def process_simulation_data(self):
        line_lengths = defaultdict(list)
        for event_number in range(1, len(self.line_log)):
            line_lengths[0].append(self.line_log[event_number]["time"])
            line_lengths[1].append(self.line_log[event_number]["line_length"])
        return line_lengths

    @staticmethod
    def calculate_position(id_counter):
        result = DataCollector.customer_line_log[id_counter][len(DataCollector.customer_line_log[id_counter])-1]["line_length"]
        return result

    def update_customer_positions(self, id_counter):
        number_of_registered_customers = len(DataCollector.customer_position_log)
        for customer_id in range(1, number_of_registered_customers+1):
            counter = DataCollector.customer_position_log[customer_id][1]['id_counter']
            if counter == id_counter:
                last_position = DataCollector.customer_position_log[customer_id][len(DataCollector.customer_position_log[customer_id])-1]["position"]
                if last_position >= 0:
                    DataCollector.customer_position_log[customer_id].append({'time': self.env.now, 'position': last_position - 1})

    @staticmethod
    def order_data_by_time():
        time_of_existance = []
        timesteps = []
        time_data = []
        counter_data = []   # save the counter_id
        position_data = []  # save the position
        amount_of_customers = len(DataCollector.customer_position_log)
        for customer_id in range(1, amount_of_customers+1):
            created = DataCollector.customer_position_log[customer_id][1]["time"]
            done = DataCollector.customer_log[len(DataCollector.customer_position_log[customer_id])-1]["time"]
            time_of_existance.append([created, done])

            number_of_events = len(DataCollector.customer_position_log[customer_id])-1
            for event in range(1, number_of_events):
                timestep = DataCollector.customer_position_log[customer_id][event]["time"]
                if timestep not in timesteps:
                    timesteps.append(timestep)

        for timestep in timesteps:
            for id_customer, lifespan in enumerate(time_of_existance):
                if lifespan[0] <= timestep and lifespan[1] >= timestep:
                    time_data.append(timestep)

    @staticmethod
    def get_shortest_queue_number(number_of_queues):
        min_queue_length = DataCollector.customer_line_log[1][len(DataCollector.customer_line_log[1])-1]["line_length"]
        shortest_queue = 1
        for line_id in range(1, number_of_queues+1):
            line_length = DataCollector.customer_line_log[line_id][len(DataCollector.customer_line_log[line_id])-1]["line_length"]
            if line_length < min_queue_length:
                shortest_queue = line_id
                min_queue_length = line_length

        return shortest_queue -1

    @staticmethod
    def visualize_sim_data(dcs):
        counter = 0
        line_lengths = []
        for dc in dcs:
            line_lengths.append(dc.process_simulation_data())
            counter += 1
        data_visualization.plot_line_length(line_lengths)
        data_storage.plott_data = line_lengths
        #dashboard.save_data_in_dashboard(line_lengths)

    @staticmethod
    def save_simulation_data_global(dcs):
        counter = 0
        line_lengths = []
        for dc in dcs:
            line_lengths.append(dc.process_simulation_data())
            counter += 1
        data_storage.plott_data = line_lengths

    @staticmethod
    def save_simulation_data(dcs):
        counter = 0
        line_lengths = []
        for dc in dcs:
            line_lengths.append(dc.process_simulation_data())
            counter += 1
        with open('output/customer_log.json', 'w') as outfile:
            json.dump({
                "customer_log": DataCollector.customer_log,
            }, outfile)
        with open('output/line_log.json', 'w') as outfile:
            for dc in dcs:
                json.dump({"line_log": dc.line_log, }, outfile)
                outfile.write("\r\n")
        with open('output/line_data.json', 'w') as outfile:
            json.dump(line_lengths, outfile)

    @staticmethod
    def merged_results(dcs):
        sum_queue_length = 0
        for d in dcs:
            sum_queue_length += d.result_line()
        avg_waiting_time = 0
        avg_time_server = 0
        for id_c in range(len(DataCollector.customer_log)):
            created = 0
            start = 0
            end = 0
            for event in range(1, len(DataCollector.customer_log[id_c])):
                if (DataCollector.customer_log[id_c][event]["type"] == "created"):
                    created = DataCollector.customer_log[id_c][event]["time"]
                elif (DataCollector.customer_log[id_c][event]["type"] == "start_scanning"):
                    start = DataCollector.customer_log[id_c][event]["time"]
                elif (DataCollector.customer_log[id_c][event]["type"] == "ended_paying"):
                    end = DataCollector.customer_log[id_c][event]["time"]
            avg_waiting_time += start - created
            avg_time_server += end - start
        avg_queue_length = sum_queue_length / len(dcs)
        avg_waiting_time = avg_waiting_time / len(DataCollector.customer_log)
        avg_time_server = avg_time_server / len(DataCollector.customer_log)
        avg_time_system = avg_waiting_time + avg_time_server
        logging.debug(f"Mittlere Wartezeit: {round(avg_waiting_time, 2)}")
        logging.debug(f"Mittlere Bedienzeit: {round(avg_time_server, 2)}")
        logging.debug(f"Mittlere Zeit im System: {round(avg_time_system, 2)}")
        logging.debug(f"Mittlere Schlangenlänge: {round(avg_queue_length, 2)}")
        logging.debug(f"Summierte Schlangenlänge: {round(sum_queue_length, 2)}")

def test_scenario():
    number_of_runs = 1
    sim_duration = 600
    onequeue = True
    number_of_checkout_points = 4
    prob_cust_spon = 0.3
    prob_cust_reg = 0.4
    prob_cust_stock = 0.3
    distribution_type = poisson
    distribution_value = 30
    data_visualization = None
    queue_desicion = "shortest"
    for _ in range(number_of_runs):
        my_c_model = CheckoutModel(sim_duration, onequeue, number_of_checkout_points,
                                   prob_cust_spon, prob_cust_reg, prob_cust_stock, distribution_type,
                                   distribution_value, data_visualization, queue_desicion)
        my_c_model.run()
        logging.debug()

def scenario_a():
    number_of_runs = 1
    sim_duration = 10800  # Zeit in Sekunden(3h), in welcher Kunden kommen
    onequeue = False
    number_of_checkout_points = 30
    prob_cust_spon = 0.2
    prob_cust_reg = 0.4
    prob_cust_stock = 0.4
    distribution_type = "exponential"
    distribution_value = 6
    data_visualization = "matplotlib"
    queue_desicion = "shortest"

    my_c_model = CheckoutModel(sim_duration, onequeue, number_of_checkout_points,
                                   prob_cust_spon, prob_cust_reg, prob_cust_stock, distribution_type,
                                   distribution_value, data_visualization, queue_desicion)
    my_c_model.run()

def dashboard_scenario(sim_duration, onequeue, number_of_checkout_points,prob_cust_spon, prob_cust_reg,
                       prob_cust_stock, distribution_type, distribution_value, data_visualization, queue_desicion):

    my_c_model = CheckoutModel(sim_duration, onequeue, number_of_checkout_points,
                               prob_cust_spon, prob_cust_reg, prob_cust_stock, distribution_type,
                               distribution_value, data_visualization, queue_desicion)
    my_c_model.run()
    print(f"Länge von plott_data: {len(data_storage.plott_data)}")
    return data_storage.plott_data


if __name__ == '__main__':
    dashboard_scenario()



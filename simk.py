import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math

# Global variable inizialization
arrival_rate = 2
departure_rates = [3, 3, 3]  # Example departure rates for k servers
termination_condition = 100
confidence_level = 0.95
num_servers = len(departure_rates)
utilization_list = []
response_list = []
expected_queue_lengths = []
throughput_list = []

# Simulating the M/M/k queue
for i in range(100):
    calendar = []  # list of tuples (event, time)
    calendar_history = []
    queue_distribution = {}
    server_states = ["free"] * num_servers
    server_free_time = 0 #Idle time
    clock = 0
    n_departure = 0
    queue_length = 0

    # utilization variables
    switch_state = 0 #Time stamp of when the server switches state

    # waiting/response time variables
    waiting_time = []
    response_time = []

    # handling first arrival
    new_event = ("arrival", 0)
    calendar.append(new_event)  # [arrival]
    calendar_history.append(new_event)

    # queue distribution has a structure made of: 
    # 1. arrival time to that queue distribution state
    # 2. time of the last event
    # 3. departure time from that queue distribution state

    queue_distribution[0] = [0, clock, 0]
        
    while n_departure < termination_condition:
        current_event = calendar.pop(0)
        clock = current_event[1]
        if current_event[0] == "arrival":
            new_event = ("arrival", clock + np.random.exponential(1/arrival_rate))
            calendar.append(new_event)  # [arrival]
            calendar_history.append(new_event)
                
            #with enumerate I generate pairs of index and state for each server
            #I filter the free servers with if state == "free"
            #with i I collect the indices of the free servers into a list
            free_servers = [i for i, state in enumerate(server_states) if state == "free"]
            if free_servers:
                chosen_server = random.choice(free_servers)
                server_states[chosen_server] = "busy"
                # if no customer is currently served the time the system is free and the free-time has to be increased
                if(len(free_servers) == num_servers):
                    server_free_time += clock - switch_state
                new_event = ("departure", clock + np.random.exponential(1/departure_rates[chosen_server]), chosen_server)
                calendar.append(new_event)  # [arrival, departure]
                calendar_history.append(new_event)
            else:
                queue_length += 1
                if queue_distribution.get(queue_length):
                    queue_distribution[queue_length][1] = clock
                    queue_distribution[queue_length-1][2] = clock
                    queue_distribution[queue_length-1][0] += clock - queue_distribution[queue_length-1][1]
                else:
                    queue_distribution[queue_length] = [0, clock, 0]
                    queue_distribution[queue_length][1] = clock
                    queue_distribution[queue_length-1][2] = clock
                    queue_distribution[queue_length-1][0] += clock - queue_distribution[queue_length-1][1]

            calendar = sorted(calendar, key=lambda clock: clock[1])

        else:  # handling of a departure
            server_states[current_event[2]] = "free"
            switch_state = current_event[1]
            n_departure += 1
            if queue_length != 0:
                queue_length -= 1
                queue_distribution[queue_length][1] = clock
                queue_distribution[queue_length+1][2] = clock
                queue_distribution[queue_length+1][0] += queue_distribution[queue_length+1][2]-queue_distribution[queue_length+1][1]
                    
                free_servers = [i for i, state in enumerate(server_states) if state == "free"]
                chosen_server = random.choice(free_servers)
                server_states[chosen_server] = "busy"
                server_free_time += clock - switch_state
                new_event = ("departure", clock + np.random.exponential(1/departure_rates[chosen_server]), chosen_server)
                calendar.append(new_event)
                calendar_history.append(new_event)
                calendar = sorted(calendar, key=lambda clock: clock[1])
    print(server_free_time)
    end_time = clock

    utilization_list.append(1-(server_free_time/end_time))

    # expected waiting time
    calendar_history = sorted(calendar_history, key=lambda clock: clock[1])
    arrivals = []
    departures = []
    for elements in calendar_history:
        if elements[0] == "arrival":
            arrivals.append(elements)
        else:
            departures.append(elements)

    waiting_time.append(0)
    for i in range(0, termination_condition-1):
        waiting = departures[i][1]-arrivals[i+1][1]
        response = departures[i][1]-arrivals[i][1]
        if waiting < 0:
            waiting = 0
        waiting_time.append(waiting)
        response_time.append(response)

    i = 0
    summatory = 0
    tn = end_time

    for elements in queue_distribution:
        summatory += i * queue_distribution[elements][0]
        i += 1

    expected_queue_lengths.append(summatory / tn )
       
    response_list.append(sum(response_time) / len(response_time))
       
    throughput_list.append(n_departure / end_time)

def calculate_confidence_interval(data):
    for i in range(len(departure_rates)):
        mean_value = float(sum(data) / len(data))
        std_value = float(stats.sem(data))
        margin_of_error = float(std_value *  stats.t.ppf((1 + confidence_level) / 2, len(data) - 1))
        ci_lower_bound = mean_value - margin_of_error
        ci_upper_bound = mean_value + margin_of_error
    return mean_value, ci_lower_bound, ci_upper_bound

#I collect the utilization values for server i across all 100 simulations
#I compute the mean and confidence interval for these values
#I use zip to transpose the list of tuples returned by "calculate condifence interval"
#into 3 separate lists: means, lower bound and upper bound

utilization_mean, utilization_lower_list, utilization_upper_list = calculate_confidence_interval(utilization_list)

response_time_mean, response_time_ci_lower, response_time_ci_upper = calculate_confidence_interval(response_list)

expected_queue_length_mean, expected_queue_length_ci_lower, expected_queue_length_ci_upper = calculate_confidence_interval(expected_queue_lengths)

throughput_mean, throughput_ci_lower, throughput_ci_upper = calculate_confidence_interval(throughput_list)

# Theoretical Values stands only if Departure Rate is higher than Arrival Rate (Convergence)


mu = departure_rates[0]  # Departure rate for each server (I consider them all the same so I take the first one)
k = len(departure_rates)  # number of servers

ro_sum = arrival_rate / mu
ro = ro_sum / k  # Usage for server


# Probability of not having clients in the system
sum_terms = sum((ro_sum)**j / math.factorial(j) for j in range(k))
last_term = (ro_sum**k) / (math.factorial(k) * (1 - ro))
print(sum_terms)
print(last_term)
P0 = 1 / (sum_terms + last_term)

# Mean queue length
Lq = (ro_sum**k / math.factorial(k)) * P0 * (ro / k) / ((1 - ro)**2)

# mean system length
L = Lq + ro_sum

# Mean waiting time in queue
Wq = Lq / arrival_rate

# Mean response time
W = Wq + 1 / mu

# System utilization
theoretical_utilization = 1 - P0

theoretical_queue_length = Lq
theoretical_response_time = W



def plot_measures(measure, xlabel, ylabel, title, ci_lower, ci_upper, little_law_value=0):
    plt.plot(measure)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(sum(measure)/len(measure), color='r',
                linestyle='--', label='Average Value')
    plt.axhline(ci_lower, color='b',
                linestyle='--', label='Lower')
    plt.axhline(ci_upper, color='y',
                linestyle='--', label='Upper')
    if little_law_value != 0:
        plt.axhline(y=little_law_value, color='purple',
                    linestyle='--', linewidth=1, label="Theoretical Value")
    plt.legend()
    plt.show()



# Print results
# utilization

print("Utilization:", sum(utilization_list)/len(utilization_list))
plot_measures(utilization_list, "Simulations", "Utilization",
              "Utilization during Simulations", utilization_lower_list,
              utilization_upper_list, theoretical_utilization)

# expected queue length

print("Expected Queue Length:", sum(
    expected_queue_lengths)/len(expected_queue_lengths))
plot_measures(expected_queue_lengths, "Simulations", "Queue Length",
              "Expected Queue Length", expected_queue_length_ci_lower,
              expected_queue_length_ci_upper, theoretical_queue_length)

# response time

print("Response Time:", sum(
    response_list)/len(response_list))
plot_measures(response_list, "Simulations", "Response",
              "Response Time during Simulations", response_time_ci_lower,
              response_time_ci_upper, theoretical_response_time)

# Throughput

print("Throughput:", sum(
    throughput_list)/len(throughput_list))
plot_measures(throughput_list, "Simulations", "Response",
              "Response Time during Simulations", throughput_ci_lower,
              throughput_ci_upper, arrival_rate)

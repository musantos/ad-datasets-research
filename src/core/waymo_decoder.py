from waymo_open_dataset.protos import scenario_pb2

def parse_waymo_scenario(serialized_data):
    """
    Decodes a binary scenario (Scenario proto) from Waymo Motion.
    Returns a Scenario object with all attributes accessible.
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytearray(serialized_data.numpy()))
    return scenario

if __name__ == "__main__":
    print("OK: Scenario decoder loaded.")
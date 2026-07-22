from waymo_open_dataset.protos import scenario_pb2

def parse_waymo_scenario(serialized_data):
    """
    Decodifica um cenario binario (Scenario proto) do Waymo Motion.
    Retorna um objeto Scenario com todos os atributos acessiveis.
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytearray(serialized_data.numpy()))
    return scenario

if __name__ == "__main__":
    print("OK: Decodificador de Scenarios carregado.")



"""An example library for converting temperatures."""


def convert_f_to_c(temperature_f):
    """Convert Fahrenheit to Celsius."""
    temperature_c = (temperature_f - 32) * (5/9)
    return temperature_c


def convert_c_to_f(temperature_c):
    """Convert Celsius to Fahrenheit."""
    temperature_f = temperature_c * (9/5) +32
    return temperature_f

def convert_f_to_k(temperature_f):
    """Convert Fahrenheit to Kelvin"""
    temperature_k = (temperature_f - 32) * (5/9) + 273.15
    return temperature_k

def convert_c_to_k(temperature_c):
    """Convert Celcius to Kelvin:"""
    temperature_k = temperature_c + 273.15 
    return temperature_k


def convert_k_to_c(temperature_k):
    """Convert Kelvin to Celsius"""
    #0K − 273.15 = -273.1°C

    temperature_c = temperature_k - 273.15 
    return temperature_c


def convert_k_to_f(temperature_k):
    """Convert Kelvin to Fahrenheit:"""

    # (0K − 273.15) × 9/5 + 32 = -459.7°F

    temperature_f = (temperature_k - 273.15)*9/5+32
    return temperature_f


def convert_f_to_all(temperature_f):
    """Give all conversions from Fahrenheit."""
    print("the temperature ", temperature_f, "degrees F is:")
    c = convert_f_to_c(temperature_f)
    k = convert_f_to_k(temperature_f)
    print("\t ",c," degrees celsius")
    print("\t ",k," degrees kelvin")
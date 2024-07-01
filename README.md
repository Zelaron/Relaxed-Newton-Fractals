# Relaxed Newton Fractals

This Python script generates Newton fractals based on a user-defined polynomial. It utilizes the relaxed Newton method, allowing for exploration of fractal patterns beyond the standard Newton-Raphson iteration.

## Features

- Generates Newton fractals for arbitrary polynomials
- Implements the relaxed Newton method with adjustable real or complex relaxation parameter
- Automatically determines appropriate plot range based on root locations
- Applies Gaussian smoothing for enhanced visual appeal
- Optionally labels roots on the output image
- Saves output as PNG files

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- SymPy

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Clone this repository or download `newton.py`.
3. Install the required packages using pip:

```pip install numpy matplotlib scipy sympy```

## Usage

1. Navigate to the directory containing `newton.py`.
2. Modify the `P(z)` function in the script to define your desired polynomial.
3. Adjust parameters such as image dimensions, maximum iterations, and relaxation parameter as needed.
4. Run the script:

```python newton.py```

5. The generated fractal will be saved in the `newton-output` directory.

## Customization

- Modify the `P(z)` function to explore different polynomials
- Adjust the `a` variable to change the relaxation parameter (default is 1)
- Set `show_root_labels = False` to hide root labels on the output image
- Modify `width` and `height` variables to change the output image dimensions

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

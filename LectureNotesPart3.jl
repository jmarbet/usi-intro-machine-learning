### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab1cffca-efd5-11ec-1d46-67a055f03758
begin
	using PlutoUI
	using ImageShow
	using ImageIO
	using FileIO
	using Plots
	using LaTeXStrings
	using LinearAlgebra
end;

# ╔═╡ d4f1cd0b-a745-4463-a59a-68b8f7b4beee
begin
	using Flux # Machine learning library
	using MLDatasets # Contains many ML datasets
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true # This is only needed in Pluto.jl to automatically download the dataset
	using Statistics# Provides basic functions such as mean()
	using ImageInTerminal # Allows us to display images in the terminal
end

# ╔═╡ 186958e8-dbd1-4bbf-94d1-10f9de56b083
md"# Beyond the Hype: An Introduction to Machine Learning and Neural Networks

Joël Marbet, June 29, 2023

This Pluto notebook aims to briefly introduce you to Julia and show you how to implement neural networks in Flux.jl. It represents \"Part 3: Putting It Into Action\" of the short course prepared for the CEMFI Undergraduate Internship 2023. 

"

# ╔═╡ 1ddf5456-0cdf-481b-a378-6fad6482ccff
md"""## A Very Brief Introduction to Julia

We will start from the basics and assume little to no background on scientific computing. For example, concepts such as variables, loops, and functions will be introduced. We will only be able to scrape the surface but I will provide references for further learning at the end.
"""

# ╔═╡ 99a749a1-4b2a-40e9-b06e-71e378e44fc2
md"""### Why Julia?

Why not learn Matlab, Python, R, Fortran, C or something else? After all, these language have very large communities and maybe you even know one of these languages already. The main reasons for learning Julia are

* High-level + High-performance
  * Fast languages like C and Fortran are "difficult" to use (e.g. they require careful specification of variable types and function inputs)
  * "Easy to use" languages like Matlab, Python and R are slow to execute
  * Julia combines fast execution time with ease of use
* Geared towards scientific computing
* Completely free and open source
* Quickly spreading over professional and scientific communities

Therefore, the purpose of using Julia goes beyond applying what we have learned so far. It might have been better to use Python from a pure machine learning perspective since that is the dominant language used in the field, but depending on your research interests (e.g., if you are interested in quantitative macroeconomics), knowing Julia will be very important.

"""

# ╔═╡ d2343252-6f80-4a29-9b70-11b323c9deec
md"###  Julia Micro-Benchmarks"

# ╔═╡ c4bcecbb-b068-4ff1-8b1b-27cfe50a7b23
load("Figures/benchmarks.png")

# ╔═╡ e16c0130-ee3f-43eb-9f78-73788a51cf7c
md"""### Installation

You can install Julia ([https://julialang.org/downloads/](https://julialang.org/downloads/)) and use any text editor you like. However, VSCode with the Julia extension offers many IDE features that improve the usability a lot. For example, with VSCode you get
  * a debugger, 
  * a profiler,
  * a list of variables in the workspace (similar to Matlab),
  * git-repository integration,
  * easy access to the Julia documentation,
  * many more features that can be added using VSCode extensions

#### Basic Installation Steps
  * Install Julia: [https://julialang.org/downloads/](https://julialang.org/downloads/)
  * Install VSCode: [https://code.visualstudio.com](https://code.visualstudio.com)
  * Install Julia for VSCode: Go to "View" in VSCode, then click on "Extensions" and type \"julia\" in the search box and hit enter. Install the Julia extension.
  * Julia packages can then be installed using Julia's package manager if necessary
"""

# ╔═╡ f927d9f6-c951-47b5-86f2-151b8f9cb555
md"### Development Environment: Julia for VSCode
	
Once everything is installed VSCode should look similar to this
"

# ╔═╡ 06e11828-2c21-4d03-b598-05f69cebcd51
load("Figures/vscode.png")

# ╔═╡ eaa545df-d2bb-4151-9430-4ad67011602d
md"""### Interacting with Julia

#### The Command-Line: REPL
"""

# ╔═╡ a108df01-11d0-4bd0-ba5b-7f752649ad68
load("Figures/REPL_white_new.png")

# ╔═╡ 995ddb34-a779-40a9-be54-2f159d862e8a
md"""
The REPL (read-eval-print loop) is the main way to interact with Julia. Here you can, for example, execute Julia code and update packages. To start this notebook, you had to run a command in REPL already.
"""			

# ╔═╡ fa070ccb-0751-4f31-b130-46ded46cf83d
"""
Different REPL modes can be activated by typing the characters below</li>
<ul>
  <li><span style="color:grey">?</span>: Shows basic documentation for functions (Prompt: <span style="color:yellow">help?</span>)</li>
  <li><span style="color:grey">]</span>: Package manager (Prompt: <span style="color:grey">(@v1.7) pkg></span>)</li>
  <li><span style="color:grey">;</span>: Execute system commands (Prompt:  <span style="color:green">shell></span>)</li>
  <li>Pressing "backspace" brings you  back to the default "Julian" mode (Prompt:  <span style="color:green">julia></span>)</li>
</ul>
""" |> HTML

# ╔═╡ 66de4aac-9738-4729-aaa0-c3e5491272dd
md"""
We will have a closer look at the REPL, once we discuss the applications towards the end of the course.
"""

# ╔═╡ f3813785-ad23-4b5d-942a-ee6c4ca1b377
md"""
#### Pluto Notebooks

At the beginning we will mainly work within this Pluto notebook. Pluto notebooks are similar to Juypter notebooks with which you might already familiar with if you know Python. However, there are some important differences.

Pluto notebooks provide a reactive interface: when one variable or function is changed, the whole notebook is updated. This can be convenient for the purpose of exploration but it puts some restictions on the code that can be written in the cells such as
* A cell can only contain a single statement
* A variable can only be defined once in a notebook

For this reason, we sometimes encapsulate the multiple lines of code in a `begin` or `let` statement to work around these restrictions. In REPL or in a Julia file, this would not be necessary.

Note that cells can be executed by pressing `shift-enter` or the run button in the bottom right of a cell.
"""

# ╔═╡ 0faed82c-589b-44a1-9ea4-82e3bab492a5
md"""
!!! note "Difference between let and begin" 
    Note that variables created in a `let` block will not be available outside that block, while variables created in a `begin` block will be available outside that block.
"""

# ╔═╡ 1509eff9-3ab2-4dfe-a78f-aff3754da8d8
md"""
### Basics

Let's start with the basics of any programming language. All programs consist of the following
  * Variables,
  * Functions,
  * Loops,
  * Conditionals

We will have a particular look on how Julia implements these concepts.

"""

# ╔═╡ 16033d7b-e4aa-47f4-a79a-897f2599ae4c
md"""
### Variables and Types

Variables are basic elements of any programming language. They
  * store information,
  * can be manipulated by the program, and
  * can be of different types, e.g. integers, floating point numbers (floats), strings (sequences of characters), or booleans (`true` or `false`)

Julia is dynamically typed, which means
  * it can infer the type of a variable from its assigned value, and
  * a programmer does not need to explicitly specify the type

For example, assigning a value of `1` to a variable called `myvar` can be done as follows
"""

# ╔═╡ 2b6cda5e-442c-4880-820b-a58d333ba307
x = 1

# ╔═╡ edf7dfc5-8325-418b-b3cd-23f568f061d1
typeof(x)

# ╔═╡ 8d0125b6-4c83-43e2-8e69-86ac69d5e511
md"""
Note that the variable name to which we assign a value is always on the left of the equal sign, while the value itself is (computed) on the right. In this case, we have created a variable of type Integer (`Int64`). Some additional examples
"""

# ╔═╡ 97b45c4f-b527-4b07-99e6-b3c6c76c58f5
let 
	a = 1			# a is of type Integer
	b = 2.0			# b is of type Float 
	c = "Hello!"	# c is of type String
	d = true		# d is of type Bool
end;

# ╔═╡ 353435cb-2f85-475f-93ba-eb1d170b2956
md"""
 Everything after `#` is a "comment" and is ignored by Julia.
"""

# ╔═╡ abc20cde-f4bd-41ef-bcc1-60371e99829d
md"""
#### Mathematical Operations

Mathematical operations can be performed as you would expect

"""

# ╔═╡ 05291062-9836-4f9b-a4b1-8a5e88ae17b3
let 
	x = 5.0			# Assign to variable x
	y = 2.0			# Assign to variable y
	z = x + y		# Perform addition
	z = x - y		# Perform subtraction
	z = x * y		# Perform multiplication
	z = x / y		# Perform division
	z = x ^ y		# Raise x to the yth power
	z = x % y		# Remainder, equivalent to rem(x,y)
end;

# ╔═╡ f5bab507-5a8f-4248-83f3-5556b2fbf771
md"""Each line above uses variables `x` and `y` to perform an operation, and assign the result to variable `z`. These operators can have alternative meanings if they are applied to variables other than integers and floats. For example, `*` can be used to join two strings 
"""

# ╔═╡ 8962a0a1-f4ce-42ec-bb2e-5439f2580ff8
"Hello " * "World"

# ╔═╡ 0f2c526c-e3b9-403b-994e-a25caa721c6c
md"""
#### Variable Names

A valid variable name must satisfy the following
  * First character: letter (a-z; A-Z), an underscore, or a subset of unicode characters (for details see the documentation)	
  * After the first char.: numbers (0-9), !, and few others are allowed

Note that some names are reserved (e.g. `if` and `function`). Such names cannot be used as variable names. There are also some stylistic conventions
  * Write variable names in lowercase letters: `myvariablename`
  * Use underscores if the variable name would be hard to read otherwise: `hard_to_read_variable_name`
  * Personally, I like to use camelcase: `myVariableName`. However, this is not proper Julia style

* Use descriptive variable names not just letters and symbols
			
"""

# ╔═╡ fc4b6ada-756d-4ca1-b864-94a1de74373b
md"""
##### Unicode Symbols

Julia is fully unciode compatible and any symbol can be used as variable or function names. In VSCode, Pluto, and REPL you can type unicode characters by typing the corresponding latex command and pressing tab key.

For example: \alpha + tab becomes `α`

"""

# ╔═╡ cc3d3ffc-f7eb-422a-9427-54112aacf53c
β = 5

# ╔═╡ a215ab12-b332-4922-9011-c5b53610f5c3
😄 = 2.0

# ╔═╡ f12771e0-7898-4761-abfd-a567c2cf7b9f
md"""
#### Arrays

A convenient way to represent large amounts of data in code are arrays. They can have an arbitrary number of dimensions
  * A one-dimensional array is also called a vector:  `x = [1, 2, 3]`

    Mathematically, we would write this as $x=\begin{pmatrix}1 \\2\\3\\\end{pmatrix}$
  * A two-dimensional array is also called a matrix: `x = [1 2; 3 4]`

    Mathematically, we would write this as $x=\begin{pmatrix}1 & 2\\3 & 4\\\end{pmatrix}$
  * An $N$-dimensional array is most easily constructed using loops

"""

# ╔═╡ 4106b5ca-8a2d-4ce3-8f97-99ee35235bf2
md"""Accessing the elements of an array can be done in multiple ways. Consider the following matrix"""

# ╔═╡ 3eb37f9e-74d9-492d-83a5-b84e6ac9278a
X = [1 2; 3 4]

# ╔═╡ 25095605-224a-4888-bf29-75b663548c10
X[1, 2] # element in the first row and second column

# ╔═╡ 715b3217-5779-4047-8c6a-8ce6162a0c00
X[2] # second element when all columns are stacked on top of each other

# ╔═╡ c4f287ea-47f3-437a-860d-874f6885d3fe
md"""The way you access arrays is quite important for speed when you loop over arrays (Linear indexing `X[2]` is faster, in general, because it takes into account how the arrays are represented in memory). 

However, linear indexing is not suitable in some applications."""

# ╔═╡ 076c20fa-b6fa-4754-a864-3643a2ba2abe
md"""

#### Mathematical Operations on Arrays

Mathematical operations are similar to those for primitive types. However, `*` does matrix multiplication for 2 dimensional arrays.
"""

# ╔═╡ ae2c0401-baa2-49b4-885b-6331a52c7069
[1 2; 3 4] * [2 3; 4 5]

# ╔═╡ aa7de575-4f0c-4720-b659-e87a6ccf1b09
md"""
Note if you need to transpose a matrix or a vector you can use `x'`.
"""

# ╔═╡ 3c7c4cf8-1689-4ddc-9258-efd80c610be1
[1 2; 3 4]'

# ╔═╡ 6107cc11-06b7-47d8-ab8d-662a3d5dc6bb
md"""Elementwise operations are possible by using the dot-notation e.g., `.*`"""

# ╔═╡ c51b7a96-46d8-4852-903f-364cfe4d46c9
[1 2; 3 4] .* [2 3; 4 5]

# ╔═╡ 9ee9c663-a236-4f0c-9ec2-b5278ffafcbe
[1 2; 3 4] .+ [2 3; 4 5]

# ╔═╡ 679d54d8-0334-4f4e-a24b-d0fbf0b98998
[1 2; 3 4] ./ [2 3; 4 5]

# ╔═╡ 0e0af689-56bc-4431-9daf-8c0106a94d81
[1 2; 3 4] .^2

# ╔═╡ 1b7be082-3e39-4497-ad0e-7ea0c96f19f1
md"""Generally, dots are used to apply operations or even functions elementwise. For example"""

# ╔═╡ 9d94fde8-135c-4ad4-9899-1704371ef0de
log.([1 2; 3 4])

# ╔═╡ b284a912-319d-416f-9093-5d3eb7233276
md"""
### Functions

Functions return some output for given inputs. They are very useful to run small code sections that you need repeatedly in different parts of you program. 

The following defines a function that adds 2 to the input and returns the result
"""

# ╔═╡ ce099828-4870-4fbf-a754-d1a9c99969b8
function h(x)
	x + 2
end

# ╔═╡ 7e857775-2eac-4ce4-8137-75ccd9546107
md"""Simple functions as the one above could also be defined as follows"""

# ╔═╡ e5e9f0c9-a7fc-4820-a575-f22cc5b2775a
h2(x) = x + 2

# ╔═╡ 16a27687-ca90-4736-9079-3bed2b40a096
md"""So, the function received input 4, added 2 and returned 6."""

# ╔═╡ 8b196d4e-4367-4a22-8e7b-8be601517ef3
md"""Functions can also have multiple inputs and outputs"""

# ╔═╡ ff385589-a034-4eab-be88-e74921164c67
function h(x, y)
	a = x + y
	b = x - y
	return a, b
end

# ╔═╡ 53fe4346-0a75-4846-b99b-1d4b863a8dc8
Markdown.parse("""
Once we have executed the code above, h(x) can be called in REPL or anywhere in your program. For example, in REPL we would get the following output

    julia>h(4)
    $(h(4))

Or in this Pluto notebook, we get
""")

# ╔═╡ 16555410-c22f-4726-af91-72c286a02897
h(4)

# ╔═╡ 198cb590-2f8a-4df5-9d6f-efc6055cf4e0
md"""Note here we explicitly used the `return` keyword. If no `return` keyword is used, the function returns the result of the last line. If we would omit `return a, b`, it would return `b` only."""

# ╔═╡ e2b13229-ab29-462e-bbd2-fefb3886a702
md"""
### If-else Statements

If-else statements are important to execute different parts of your program depending on whether certain conditions are met. They can have a single branch (this example) or multiple (see below)
"""

# ╔═╡ 00b5e789-e62e-440c-89c6-a26fd71633bc
begin
	julia_is_cool = true
	if julia_is_cool
		println("It sure is cool!")
	end
end

# ╔═╡ 9c2c40d5-1938-4a3a-828d-d1f5fed19618
md"""
The code in a branch is only evaluated if the condition is `true`. The following code checks the sign of `x` and prints the result (in REPL).

Note that further up in the notebook we have defined x = $(x).
"""

# ╔═╡ a15260d5-3ef4-4a67-8512-810a2b76de49
if x < 0
	println("x is negative")
elseif x > 0
	println("x is positive")
else
	println("x is zero")
end

# ╔═╡ a73e8399-ba89-4ccb-b2a1-93c34d788509
md"""
Here is a short overview of comparisons that can be made
   
    ==		# equality
    !=, ≠	# inequality
    <		# less than
    <=, ≤	# less than or equal to
    >		# greater than
    >=, ≥	# greater than or equal to

Chaining comparisons is also possible
"""

# ╔═╡ 191fee5a-5e6e-4b91-b5f3-9919933e808e
if 0 < x < 3
	println("x is between 0 and 3")
end

# ╔═╡ 6fdac8e1-286b-46a3-a563-6fa5cca48c09
md"""
We can also make multiple comparisons at the same time
"""

# ╔═╡ 9eba8bcb-7d99-4f78-a519-86a5dfd08035
if 0 < x  && x != 3
	println("x is positive and not 3")
elseif x == -4  || x == -5
	println("x is is -4 or -5")
end

# ╔═╡ 3ea37951-0d48-4110-bd80-ab9afed7ce37
md"""
### Loops

Loops repeat the same operations several times. Julia offers `for` loops and `while` loops
  * `for` loops go over each element of an iterable object (i.e. a range such as `1:10` or `["Julia", "Python", "Matlab"]`)
  * `while` loops execute the same code until the condition for their execution is not true anymore
  * Ultimately, both can be used to achieve the same results

The following goes over all integers from 1 to 10 and adds them to `x`

"""

# ╔═╡ 2f0ec430-440a-4b09-99ac-73f5558de07c
let 
	x = 0
	for i in 1:10
		x = x + i
	end
	x
end

# ╔═╡ 5c6c4774-071d-42e3-b1b1-88cc55c8ce16
md"""Mathematically, this corresponds to 

$$x = \sum_{i=1}^{10} i$$"""

# ╔═╡ 7928b444-c7df-4750-9bdc-6711d7f981a5
md"""
`while` loops have a slightly different syntax

    while julia_is_cool
    	println("Yep, Julia is still cool. I just checked it.")
    end

Note the loop above would continue indefinitely.
"""

# ╔═╡ 9b02f32d-aa48-4c7f-8984-d998735e4187
md"""

### Additional Resources

* Julia Short Course 2022: [https://github.com/jmarbet/julia-shortcourse](https://github.com/jmarbet/julia-shortcourse)
* TechyTok!: [https://techytok.com/from-zero-to-julia/](https://techytok.com/from-zero-to-julia/)
  * Excellent tutorial that goes into more detail than we will be able to
* QuantEcon: [https://julia.quantecon.org/](https://julia.quantecon.org/)
  * Provides great lectures that start from the very basics of Julia
  * Many economic applications
* Julia Documentation:  [https://docs.julialang.org/](https://docs.julialang.org/)
  * Very clear and well organized
  * Performance tips: [https://docs.julialang.org/en/v1/manual/performance-tips/](https://docs.julialang.org/en/v1/manual/performance-tips/)
  * Noteworthy Differences from other Languages: [https://docs.julialang.org/en/v1/manual/noteworthy-differences/](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
    * If you have experience in either Matlab, R, Python or C/C++, it's a good idea to have a look at the respective section
* Plotting with Julia
  * Plots.jl: [http://docs.juliaplots.org/](http://docs.juliaplots.org/)

"""

# ╔═╡ 1f971adb-c94e-477f-9d82-613344455915
md"""## Implementing an Artificial Neuron From Scratch

To familiarize ourselves with Julia and improve our understanding of the basics of neural networks, it is useful to implement some parts in Julia from scratch, i.e., without using additional packages. Later on, we will implement the full neural network in Flux.jl.

Recall the artificial neuron defined in the lecture notes

"""

# ╔═╡ 5c7f0e32-b888-4305-915f-b9d5b0a53520
load("Figures/artificial_neuron.png")

# ╔═╡ 21d3f754-317a-42c3-a0ee-75a9f4a57468
md"""
Mathematically, we have that $N$ inputs $x=(x_1,x_2,\ldots,x_N)'$ are linearly combined into

$$z = b + \sum_{i=1}^N w_i x_i = \sum_{i=0}^N w_i x_i\,,$$

where we defined an additional input $x_0=1$ and $w_0=b$, and 

$$a = \phi(z) = \phi\left( \sum_{i=0}^N w_i x_i \right)\,,$$

where $\phi(\cdot)$ is the activation function.
"""

# ╔═╡ 8d75120d-08af-43bb-95cd-a429b2888986
md"""
### Activation Function

Let's start by defining an activation function. Common activation functions include
* Sigmoid: $\phi(z) = \frac{1}{1+e^{-z}}$
* Hyperbolic tangent: $\phi(z) = tanh(z)$
* Rectified linear unit (ReLU): $\phi(z) = \max(0,z)$
* Softplus: $\phi(z) =\log(1+e^{z})$


"""

# ╔═╡ 340cc332-bbb1-4e04-a0db-f4fb49280c0e
md"""
!!! note "Mini-Exercise"
	Define a function $\phi(x)$ with an activation function of your choice.

	Note that you need to use the greek symbol (type "\phi" and press tab), otherwise the function will not be plotted.

"""

# ╔═╡ 0d34b214-5e4d-4aba-af3f-180c82295254
# Define the activation function here


# ╔═╡ 9a6b0c01-182b-4466-b204-fe8aa9316658
try
	plot(-2:0.01:2, ϕ, title = L"Activation Function $\phi(x)$", legend = :none)
catch
	plot((1:3)', framestyle=:none, legend = :none)
	xlims!(0.0,2.0)
	ylims!(0.0,2.0)
	annotate!(1.0, 1.0, L"$\phi(x)$ not defined")
end

# ╔═╡ 053125c4-2c54-4509-b27a-c63cbe9cadb4
md"""
!!! hint "Mini-Exercise: Solution"
	For example, if you wanted to use ReLU you could use

	```
		ϕ(x) = max(0, x)
	```

	or

	```
		function ϕ(x)
			max(0, x)
		end
	```

	or 

	```
		function ϕ(x)
			if x < 0
				return 0
			else
				return x
			end
		end
	```
"""

# ╔═╡ f6370a67-5a05-4f5f-841b-c59171dc3ac6
md"""
### An Artificial Neuron

Given the activation function, we now want to implement our artificial neuron, i.e., we want to compute

$$a = \phi\left( b + \sum_{i=1}^N w_i x_i \right)\,,$$

"""

# ╔═╡ 19ecb197-624b-49e1-9994-f7a35c7fce07
md"""
!!! note "Mini-Exercise"
	Define a function called `neuron` that takes a vector $x$ (inputs) and vector $w$ (weights) and $b$ (bias) as inputs, and computes

	$$a = \phi\left( b + \sum_{i=1}^N w_i x_i \right)\,,$$

"""

# ╔═╡ ea0220db-574e-4ea4-b0b9-93eda5144d32
function neuron(x, w, b)
	# Implement the weighted summation and activation here
end

# ╔═╡ 009d9ffb-c37e-4bfd-8cb2-38c12bb5aee0
md"""To test the function we also have to define $x$, $w$, and $b$."""

# ╔═╡ fef1067d-91a5-4d13-ad11-4400c34fd5db
let 
	x0 = [0.1, 0.4, 1.2]
	w0 = [0.01, -0.2, 1.05]
	b0 = 0.1
	neuron(x0, w0, b0)
end

# ╔═╡ cb7629fd-cb9a-4ff0-806a-5fdb1cb55b35
md"""
!!! hint "Mini-Exercise: Solution"
	Possible solutions might look as follows

	```
		function neuron(x, w, b)
			z = dot(x,w) + b
			a = ϕ(z)
			return a
		end
	```

	or

	```
		function neuron(x, w, b)
			z = x'*w + b
			a = ϕ(z)
			return a
		end
	```

	or with a loop

	```
		function neuron(x, w, b)
			z = b
			for ii in 1:length(x)
				z += x[ii] * w[ii] + b
				# Alternatively: z = z + x[ii] * w[ii] + b
			end
			a = ϕ(z)
			return a
		end
	```
"""

# ╔═╡ 438ccc1a-88fc-435b-9d5a-bc34312b5bcf
md"""
Let's have a look at the plot of our artifical neuron if we just have two inputs $(x_1,x_2)$.
"""

# ╔═╡ 39e07c43-2d93-4745-8314-43ec69ad60ec
md"""

If everything works as expected, you can vary the weights and bias of your artificial neuron here

w1 $(@bind w1 Slider(-2.0:0.01:2.0, show_value = true, default = 1.0))

w2 $(@bind w2 Slider(-2.0:0.01:2.0, show_value = true, default = 1.0))

b $(@bind b Slider(-2.0:0.01:2.0, show_value = true, default = 1.0))

Depending on the activation function, you might want to adjust the z-axis limits of the plot

zmin = $(@bind zmin NumberField(-10:0.25:10, default=-1))
zmax = $(@bind zmax NumberField(-10:0.25:10, default=1))

"""

# ╔═╡ c45ff159-9b32-4268-b52f-8430c4e02d61
try
	surface(-2.0:0.01:2.0, -2.0:0.01:2.0, (x1, x2) -> neuron([x1, x2], [w1, w2], b), 
		cbar = :none,
		xlabel = L"x_1",
		ylabel = L"x_2",
		zlabel = L"a",
		title = "Output of Artificial Neuron", zlims = (zmin, zmax))
catch
	plot((1:3)', framestyle=:none, legend = :none)
	xlims!(0.0,2.0)
	ylims!(0.0,2.0)
	annotate!(1.0, 1.0, L"$\phi(x)$ or $neuron(x,w,b)$ not defined")
end

# ╔═╡ 6e7a8d1d-40b0-4319-a438-27eacf5fe754
md"""
## Implementing a Neural Network from Scratch

For simplicity, let's consider neural networks with a single layer only. Recall that a single-layer neural network is a linear combination of $M$ artificial neurons $a_j$
  	
$$a_j = \phi(z_j) = \phi\left( b_{j}^{1} + \sum_{i=1}^N w_{ji}^{1} x_i \right)\,,$$
  		
with the output defined as

$$g(x ; w) = b^{2}+\sum_{j=1}^{M} w_{j}^{2} a_j\,,$$

Note that it will be convenient to represent everything as vectors. We can write

$$z = b^{1} + w^1 x\,,$$

where $x$ is a $N\times 1$ vector  of inputs, $z=(z_1,z_2,\ldots,z_M)'$ is a $M\times 1$ vector, $b^1=(b_1^1,b_2^1,\ldots,b_M^1)'$ is $M\times 1$ vector of biases in hidden layer, and $w^1$ is $M\times N$ matrix of weights in the hidden layer.

Applying the activation function to each linear combination of inputs yields $a$

$$a = \phi(z)\,,$$

and combining them linearly in the output layer can be written as

$$g(x,w) = b^2 + w^2 * a\,,$$

where $b^1=(b_1^1,b_2^1,\ldots,b_K^1)'$ is $K\times 1$ vector of biases in the output layer, and $w^1$ is $K\times M$ matrix of weights in the output layer with ouptut $g(x,w)$ having $K\times 1$ elements.

"""

# ╔═╡ d5464c89-8185-4b63-b77c-e941a09dc95d
md"""
!!! note "Mini-Exercise"
	Define a function called `feedforward` that takes a vector $x$ (inputs), $w1$ (weights in hidden layer), $w2$ (weights in output layer), $b1$ (biases in hidden layer), and $b2$ (biases in output layer) as inputs, and computes

	$$z = b^{1} + w^1 x\,,$$
	$$a = \phi(z)\,,$$
	$$g = b^2 + w^2 * a\,,$$

	and returns $g$ at the end.

"""

# ╔═╡ 1d7bd4c3-70f0-4eba-9536-570fa84148c5
function feedforward(x, w1, w2, b1, b2)
	# Implement feedforward equations here
end

# ╔═╡ 0537a88a-81c4-4523-9f18-7e3f9fc4e310
md"""To test our (feedforward) neural network, we can write"""

# ╔═╡ 42deb0ec-4beb-4dda-b206-65af56866577
let 
	M = 4
	K = 2
	x0 = [0.1, 0.4, 1.2]
	w10 = randn(M, length(x0)) # Note: randn() draws random numbers from a normal distribution
	b10 = randn(M)
	w20 = randn(K, M)
	b20 = randn(K)
	feedforward(x0, w10, w20, b10, b20)
end

# ╔═╡ 785eb34f-168b-4d2f-b08b-7a531cf66fde
md"""
!!! hint "Mini-Exercise: Solution"
	A possible solution might look as follows

	```
		function feedforward(x, w1, w2, b1, b2)
			z = b1 .+ w1 * x
			a = ϕ.(z) # Note the . is needed to apply the function to each element
			g = b2 .+ w2 * a
			return g
		end
	```
"""

# ╔═╡ 8fc72595-166a-487e-9212-b6927f0f1876
md"""

If everything works as expected, you can vary the weights and bias of your neural network here

##### Hidden Layer Parameters

w11 $(@bind w11 Slider(-5.0:0.01:5.0, show_value = true, default = -3.0))

w12 $(@bind w12 Slider(-5.0:0.01:5.0, show_value = true, default = 1.0))

w13 $(@bind w13 Slider(-5.0:0.01:5.0, show_value = true, default = 3.0))

b11 $(@bind b11 Slider(-5.0:0.01:5.0, show_value = true, default = -1.5))

b12 $(@bind b12 Slider(-5.0:0.01:5.0, show_value = true, default = 1.0))

b13 $(@bind b13 Slider(-5.0:0.01:5.0, show_value = true, default = -3.0))

##### Output Layer Parameters

w21 $(@bind w21 Slider(-5.0:0.01:5.0, show_value = true, default = 1.0))

w22 $(@bind w22 Slider(-5.0:0.01:5.0, show_value = true, default = 1.0))

w23 $(@bind w23 Slider(-5.0:0.01:5.0, show_value = true, default = -1.0))

b21 $(@bind b21 Slider(-5.0:0.01:5.0, show_value = true, default = 0.0))


"""

# ╔═╡ 36016caa-c799-41bb-86f3-c57fc8739411
try
	plot(-2.0:0.01:2.0, x -> exp(x)-x^3, label = "f(x)")
	plot!(-2.0:0.01:2.0, (x1) -> feedforward([x1], [w11 w12 w13]', [w21 w22 w23], [b11, b12, b13], [b21])[1], 
		label = "g(x;w)",
		xlabel = L"x",
		title = "Output of Neural Network")
catch
	plot((1:3)', framestyle=:none, legend = :none)
	xlims!(0.0,2.0)
	ylims!(0.0,2.0)
	annotate!(1.0, 1.0, L"$\phi(x)$ and $feedforward(x,w^1,w^2,b^1,b^2)$ not defined")
end

# ╔═╡ c994cbc8-e31c-41fa-b0c2-e935c6203f99
md"""
### Determining Weights and Biases

For tractability, we used three hidden nodes $M=3$ and a single input and output $N=K=1$ in the example in the previous section. Nevertheless, this resulted in 10 parameters to choose! As you can see, doing this by hand is almost impossible. Even if we do find some parameters that somewhat work well, $M=3$ is likely not sufficient to yield a good approximation.

In practice, the weights (and biases) are determined through gradient descent, with the gradient evaluated using a backpropagation algorithm. Unfortunately, we do not have the time to implement this ourselves.

If you are interested, you can have a look at the lecture notes of "Intro to Neural Networks" which I taught at the ECB in 2021 ([https://github.com/jmarbet/intro-to-neural-networks](https://github.com/jmarbet/intro-to-neural-networks)). Furthermore, I can recommend the online book [http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com) to get an introduction to neural networks and backpropagation.

For the purpose of this introduction, we will continue in Flux.jl which provides functions to handle the training of a neural network.

"""

# ╔═╡ eebfdc77-bbe0-407e-8559-66bbfe4f07da
md"""
## Example: Function Approximation

Let's implement the neural network from the previous section in Flux.jl.

First, we need to draw random data. We will try to approximate the function 

$$f(x) = exp(x) - x^3$$

in the interval (-2, 2). For this reason, we only draw samples in that range.

"""

# ╔═╡ fd2b3bac-5bee-4f42-9ad7-c7ec15585a84
begin

	# Define the function to approximate
	f(x) = exp(x) - x^3
	
	# Sample random inputs in (-2, 2)
	N = 10000
	inputs = 4 * rand(Float32, 1, N) .- 2 # Random 1xN matrix
	  
	# Evaluate function f for each input 
	outputs = f.(inputs)

	# Note: To improve speed, make sure that inputs and outputs are of type Float32
	println("Type of inputs: ", typeof(inputs))
	println("Type of outputs: ", typeof(outputs))
	
	# Prepare DataLoader with given batchsisze
	# For stochastic gradient descent set batchsize = 1
	# For full batch gradient descent set batchsize = N
	trainingData = Flux.DataLoader((inputs, outputs); batchsize = 8, shuffle=true)

end

# ╔═╡ 83d2b40d-429e-40a4-b658-0b15c30b620b
md"""We define a neural network with a single hidden layer and `nHiddenNeuronsApprox =` $(@bind nHiddenNeuronsApprox NumberField(1:100, default=5)) hidden neurons."""

# ╔═╡ 49c8663a-b113-4655-9ce1-1eaaf99221e6
modelApprox = Chain(
	Dense(1 => nHiddenNeuronsApprox, sigmoid), 	# Input-Hidden (sigmoid activation)
	Dense(nHiddenNeuronsApprox => 1)    		# Hidden-Output (linear activation)
)

# ╔═╡ 65e9ccd3-4ba4-4c7f-9720-65629bb02caf
md"""Note that the resulting neural network has $(sum(length.(Flux.params(modelApprox)))) parameters. The neural network can then be trained as follows."""

# ╔═╡ 754c6392-a468-4dde-9ef3-eee177145237
# ╠═╡ skip_as_script = true
#=╠═╡
begin

	epochs = 200  
    opt = Flux.setup(Adam(), modelApprox) # Gradient descent with Adam optimizer
	#opt = Flux.setup(Descent(0.01), modelApprox) # Gradient descent with fixed learning rate
	trainingLosses = zeros(epochs)
    
    for epoch in 1:epochs

		loss = 0.0
		
		for (x, y) in trainingData
			
	        # Compute the loss and the gradients
	        l, gs = Flux.withgradient(m -> Flux.mse(m(x), y), modelApprox)
			
	        # Update the model parameters 
	        Flux.update!(opt, modelApprox, gs[1])
			
	        # Accumulate the mean loss
	        loss += l / length(trainingData)
			
	    end

		# Save current training loss
		trainingLosses[epoch] = loss

		if mod(epoch, 25) == 1
	        @info "After epoch $epoch" loss
	    end

    end

end
  ╠═╡ =#

# ╔═╡ e33a6bb6-81a8-430e-a755-09e809a3a9c3
#=╠═╡
plot(trainingLosses, xlabel = "Epoch", ylabel = "Loss", title = "Training Loss", legend = :none)
  ╠═╡ =#

# ╔═╡ a9447a6b-bf7c-4fa1-b501-d2957579446e
begin
	plot(-2.0:0.01:2.0, x -> exp(x)-x^3, label = "f(x)")
	plot!(-2.0f0:0.01f0:2.0f0, x -> modelApprox([x])[1], 
		label = "g(x,w)",
		xlabel = L"x",
		title = "Output of Trained Neural Network")
end

# ╔═╡ eacc1145-c301-40a9-8dd9-33d3aeb060dc
md"""
## Example: Recognition of Handwritten Digits

Note: This example is based on the Flux Model Zoo ([https://github.com/FluxML/model-zoo/](https://github.com/FluxML/model-zoo/)).

"""

# ╔═╡ e93791f9-4ea9-488c-aff3-40965480ce42
md"""
### Data Set: MNIST

We will be using the MNIST (Modified National Institute of Standards and Technology)  data set, which contains a large number of handwritten digits.

"""

# ╔═╡ 4eca1bb7-c43e-416c-b24d-488d3154b986
train_data = MLDatasets.MNIST()

# ╔═╡ e28c4c84-cfaa-4e9e-a358-015d3107dcb2
test_data = MLDatasets.MNIST(split=:test)


# ╔═╡ 1608b788-8ec5-44f4-9166-2d54e201a1c0
md"""
There are 60000 images of handwritten digits in our (training) data set (there are 10000 additional test images). Each one of them is represented by a $28\times 28$ matrix of values between 0 and 1. These values correspond to different shades of grey with 0 meaning black and 1 meaning white in  a $28\times 28$ pixel image. For example, here is digit `ndig=` $(@bind ndig NumberField(1:60000, default=4667))  in the data set
"""

# ╔═╡ a31f1a57-cb1a-4f43-98f2-5fc751f58c9f
train_data[ndig].features

# ╔═╡ 07956497-d26b-4d30-be45-bd70893062c5
md"""Or as an image"""

# ╔═╡ 51e8eca7-bc01-4e02-b3f4-a9cad92dc3fa
convert2image(train_data, train_data[ndig].features)

# ╔═╡ 7d48f962-d80d-447b-b3b8-dc4e684cf2e4
md"""
Importantly, the data set also has a label for the digit, which in this case is
"""

# ╔═╡ 7bbfdb5a-2332-412f-8157-780df408bc74
train_data[ndig].targets

# ╔═╡ 59311841-c40f-4046-82d6-ee0480252c31
md"""
### Task: Classification of Images

Our goal is to use a neural network to recognize handwritten digits in the validation dataset (i.e., in the additional 10000 test images). Essentially, we want to find a mapping from the $28\times 28$ matrix of pixels into digits from 0 to 9.

This is a form of supervised learning, more precisely a classification task.

"""

# ╔═╡ ff7c59b9-e2aa-4c75-96a2-b97eda9befd7
md"""
### Implementation in Flux.jl

#### Setting up the Neural Network
We are going to use a neural network with a single hidden layer with `nHiddenNeurons=` $(@bind nHiddenNeurons NumberField(1:100, default=32)) neurons. Let's define the neural network layers (this defines a function called `model(x)`)
   
"""

# ╔═╡ 21ad80f3-ffaf-4e85-814c-3a0609d7cc3b
model = Chain(
	Dense(28^2 => nHiddenNeurons, sigmoid), # Input-Hidden (sigmoid activation)
    Dense(nHiddenNeurons => 10),    # Hidden-Output
	softmax
)

# ╔═╡ 9269fb9d-13e6-49da-9a95-7f065c36b4d1
md"""
Note that on the output layer, we use a softmax to make sure that the outputs can be interpreted as probabilities (i.e., they are between 0 and 1, and sum up to 1). Thus, the output of the neural network will be a probability distribution over the 10 digits. Furthermore, note that the resulting neural network has $(sum(length.(Flux.params(model)))) parameters!

Feeding random data into the neural network yields

"""

# ╔═╡ 24dab50f-d8cc-4129-b34f-2d9aa522191c
p1 = model(rand(Float32, 28^2))

# ╔═╡ 9714bcc7-7dd4-471b-b781-6cf7786dd805
md"""which sums up to 1"""

# ╔═╡ b46d0235-5c0d-4439-926a-a9c10e541c48
sum(p1) ≈ 1

# ╔═╡ d02593d9-9f3a-488c-8fc3-b3fa2aff0f35
md"""as expected. We can also feed in multiple random "images" at once"""

# ╔═╡ ea7d1d69-12ff-4b58-aea7-d59f885d5e2b
 model(rand(Float32, 28^2, 3)) 

# ╔═╡ 36d38f50-9038-420c-a3d0-c318a5ab2776
md"""
#### Preparing the Data

For Flux.jl to be able to train the network using the data, the data needs to be in a particular form. In particular, we need to provide it an object of type `Flux.DataLoader`.

The following function takes care of this
"""

# ╔═╡ 4c1efec6-2499-43bc-ae85-873090fd6924
function simple_loader(data::MNIST; batchsize::Int=64)
    x2dim = reshape(data.features, 28^2, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x2dim, yhot); batchsize, shuffle=true)
end

# ╔═╡ a2fef07c-d41c-4263-bd69-011860beb824
md"""
First, it changes the shape of the $28\times 28\times 60000$ matrix containg the 60000 images into a $768\times 60000$ (i.e., one image per column) denoted `x2dim`. 

In other words, it converts this
"""

# ╔═╡ f2fa67c8-bb19-4b81-8d9d-5375d574bc39
train_data.features

# ╔═╡ d2a44c70-00ef-41f7-86c2-4af3503df3e1
md"""into this"""

# ╔═╡ dd0ebc7d-0b99-4d7c-b47b-8b44c34c9939
reshape(train_data.features, 28^2, :)

# ╔═╡ 60c8d206-528a-462b-ae3e-f684dc1a5163
md"""And furthermore, it converts the $60000\times 1$ vector of labels into a $10\times 60000$ matrix, where each column represents digits from 0 to 9, meaning that it converts this"""

# ╔═╡ 5e57095e-8962-4a0d-8993-5ddb5e2cf643
train_data.targets

# ╔═╡ 5dd3e1d5-b408-488c-8e4f-391f061a024f
md"""into this"""

# ╔═╡ 3b2980a6-7b36-4353-9ff4-41a2bc5ab6ff
Flux.onehotbatch(train_data.targets, 0:9)

# ╔═╡ 1674f0ac-e3a8-4b3b-b398-e951add574a3
md"""
Recall that this is also how the inputs and outputs were represented in our implementation, with the only difference being that we had only a single data point, i.e., only one column.

Finally, `Flux.DataLoader` splits the data set into training batches. During the training, we will make one gradient descent step for each batch.
"""

# ╔═╡ f1b83d73-39e8-4f37-82b7-eec404341c6e
md"""
Let's check whether things are working as expected. 

"""

# ╔═╡ e41a889a-8cd1-41de-a4fc-728c3ce38334
let
	x1, y1 = first(simple_loader(train_data)) # Take the first batch 
	model(x1) # Feed the first batch into the neural network
end

# ╔═╡ d5648a74-65e2-4b14-8f0f-c4eea9e188c2
md"""
#### Other Preparations

To evaluate, whether our neural network makes good predictions, we also need a function to evaluate its accuracy. The following function takes care of this
"""

# ╔═╡ f9caa6b4-2718-4fa7-bd85-5fc54d51b652
function simple_accuracy(model, data::MNIST=test_data)
    (x, y) = only(simple_loader(data; batchsize=length(data)))  # make one big batch
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)  # BitVector
    acc = round(100 * mean(iscorrect); digits=2)
end

# ╔═╡ bc229146-ef6e-49ba-8aa3-02fdad52a430
md"""
We can evaluate the accuracy of our neural network as follows (note: by default the accuracy is evaluated on the validation (or test) data set and not the training data set.)
"""

# ╔═╡ 182df020-a9e9-4d60-a579-d08d307e40f5
simple_accuracy(model)

# ╔═╡ db537cc3-09b0-4d6e-b5bb-2eb67877e258
md"""
Thus, our untrained neural network has an accuracy of around 10%, which is exactly what you would expect if one would simply guess the digits.
"""

# ╔═╡ 35944668-08b5-49ec-923f-d689f3f923b3
md"""
#### Training

All that is left is to use gradient descent to train the neural network.

We prepare the data set as described above
"""

# ╔═╡ 668d011d-cade-4e97-b35b-21190ecca084
train_loader = simple_loader(train_data, batchsize = 256)


# ╔═╡ 38d3369a-8589-4d32-b36b-174aa07d183f
md"""
Initialize the optimizer for the gradient descent.
"""

# ╔═╡ 35071e0f-b195-473a-a300-b50488d441e0
opt_state = Flux.setup(Adam(3e-4), model);


# ╔═╡ dcc595d7-ae88-420e-8b5f-057e6d1ea0e0
md"""
and finally do the gradient descent. Note that we train the model for 30 epochs (i.e., we go over the whole data set for 30 times). In each epoch, we make a number of gradient descent steps equal to the batch size.
"""

# ╔═╡ e19cf30e-03ef-42a6-88a4-04938faf9fe4
# ╠═╡ skip_as_script = true
#=╠═╡
for epoch in 1:30
	
    loss = 0.0
    
	for (x, y) in train_loader
    
		# Compute the loss and the gradients
        l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
        
		# Update the model parameters (and the Adam momenta)
        Flux.update!(opt_state, model, gs[1])
        
		# Accumulate the mean loss, just for logging
        loss += l / length(train_loader)
    
	end

    if mod(epoch, 2) == 1
		# Report on train and test, only every 2nd epoch:
        train_acc = simple_accuracy(model, train_data)
        test_acc = simple_accuracy(model, test_data)
        @info "After epoch $epoch" loss train_acc test_acc
    end

end
  ╠═╡ =#

# ╔═╡ 34af0c6c-6f68-43a6-b899-5f61f39c277f
md"""
Note that after training the accuracy of our neural network has increased to $(simple_accuracy(model, test_data))%. Hurray!

Let's have a look at some predictions of our neural network by looking at digit `nTest=` $(@bind nTest NumberField(1:10000, default=32)) in the validation data set.

"""

# ╔═╡ a95e6d5b-9996-45db-b061-1cba033b1a6c
convert2image(test_data, test_data[nTest].features) 

# ╔═╡ 5d456d2f-aa3c-47d0-b1c5-5131d08fc031
test_data[nTest].targets

# ╔═╡ 2f36f77e-7739-4d60-b849-4d9adc48bf56
model(test_data[nTest].features[:])


# ╔═╡ 4c4a3768-0afd-438f-b52c-a21c26463892
md"""
Thus, our neural network thinks that the digit shown above is a $(findmax(model(test_data[nTest].features[:]))[2]-1) with $(round(convert(Float64, findmax(model(test_data[nTest].features[:]))[1]*100), digits = 2))% probability. Pretty cool, isn't it? 
"""

# ╔═╡ 7a7deb5e-3a6a-4cba-89e4-287f04cb2ae1
md"""
!!! note "Mini-Exercise 1"
	Play around with the neural network settings. For example, you could change
	* the number of hidden neurons
	* the activation function
	* the number of hidden layers
	
"""

# ╔═╡ 58c2ed5e-4907-4358-8ee1-64b11b957be2
md"""
!!! note "Mini-Exercise 2"
	The Fashion MNIST dataset is similar to the MNIST data set but uses images of clothing instead of handwritten digits. Try to adapt the code above and train the network to recognize clothing instead of handwritten digits.

	Check out [https://juliaml.github.io/MLDatasets.jl/stable/datasets/vision/](https://juliaml.github.io/MLDatasets.jl/stable/datasets/vision/) for alternative dataset for which the code above can easily be adapted.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FileIO = "~1.16.1"
Flux = "~0.13.17"
ImageIO = "~0.6.6"
ImageInTerminal = "~0.5.2"
ImageShow = "~0.3.7"
LaTeXStrings = "~1.3.0"
MLDatasets = "~0.7.11"
Plots = "~1.38.16"
PlutoUI = "~0.7.51"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "06617ce305816b2246ab7b5c4c1d38db90e3f380"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BufferedStreams]]
git-tree-sha1 = "5bcb75a2979e40b29eb250cb26daab67aa8f97f5"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "442d989978ed3ff4e174c928ee879dc09d1ef693"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.3.2"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2918fbffb50e3b7a0b9127617587afa76d4276e8"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.8.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Chemfiles]]
deps = ["Chemfiles_jll", "DocStringExtensions"]
git-tree-sha1 = "6951fe6a535a07041122a3a6860a63a7a83e081e"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.40"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "4e88377ae7ebeaf29a047aa1ee40826e0b708a5d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.7.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "6e8d74545d34528c30ccd3fa0f3c00f8ed49584c"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.11"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "0b3b52afd0f87b0a3f5ada0466352d125c9db458"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote", "cuDNN"]
git-tree-sha1 = "3e2c3704c2173ab4b1935362384ca878b53d4c34"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.17"

    [deps.Flux.extensions]
    AMDGPUExt = "AMDGPU"
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "478f8c3145bb91d82c2cf20433e8c1b30df454cc"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "cb090aea21c6ca78d59672a7e7d13bd56d09de64"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.20.3"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "c73fdc3d9da7700691848b78c61841274076932a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.15"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2613d054b0e18a3dea99ca1594e9a3960e025da4"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.7"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageInTerminal]]
deps = ["ColorTypes", "Crayons", "FileIO", "Sixel", "XTermColors"]
git-tree-sha1 = "f260604e7600723a323b42cb92ae22d837cd5dc9"
uuid = "d8c32880-2388-543b-8c61-d9f865259254"
version = "0.5.2"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "5b62d93f2582b09e469b3099d839c2d2ebf5066d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.13.1"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "b48617c5d764908b5fac493cd907cf33cc11eec1"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "5007c1421563108110bbd57f63d8ad4565808818"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.2.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "1222116d7313cdefecf3d45a2bc1a89c4e7c9217"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.22+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "79fd0b5ee384caf8ebba6c8fb3f365ca3e2c5493"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "a03a093b03824f07fe00931df76b18d99398ebb9"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.11"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "72240e3f5ca031937bd536182cb2c031da5f46dd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.21"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6a01f65dd8583dee82eecc2a19b0ff21521aa749"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.18"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "2e71d7dbcab8dc47306c0ed6ac6018fbc1a7070f"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.3"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XTermColors]]
deps = ["Crayons", "ImageBase", "OffsetArrays"]
git-tree-sha1 = "bc27b7622a51f570c57b80bd839d1c0d43605b38"
uuid = "c8c2cc18-de81-4e68-b407-38a3a0c0491f"
version = "0.2.1"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "f65490d187861d6222cb38bcbbff3fd949a7ec3e"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.0.4"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─186958e8-dbd1-4bbf-94d1-10f9de56b083
# ╠═ab1cffca-efd5-11ec-1d46-67a055f03758
# ╟─1ddf5456-0cdf-481b-a378-6fad6482ccff
# ╟─99a749a1-4b2a-40e9-b06e-71e378e44fc2
# ╟─d2343252-6f80-4a29-9b70-11b323c9deec
# ╟─c4bcecbb-b068-4ff1-8b1b-27cfe50a7b23
# ╟─e16c0130-ee3f-43eb-9f78-73788a51cf7c
# ╟─f927d9f6-c951-47b5-86f2-151b8f9cb555
# ╟─06e11828-2c21-4d03-b598-05f69cebcd51
# ╟─eaa545df-d2bb-4151-9430-4ad67011602d
# ╟─a108df01-11d0-4bd0-ba5b-7f752649ad68
# ╟─995ddb34-a779-40a9-be54-2f159d862e8a
# ╟─fa070ccb-0751-4f31-b130-46ded46cf83d
# ╟─66de4aac-9738-4729-aaa0-c3e5491272dd
# ╟─f3813785-ad23-4b5d-942a-ee6c4ca1b377
# ╟─0faed82c-589b-44a1-9ea4-82e3bab492a5
# ╟─1509eff9-3ab2-4dfe-a78f-aff3754da8d8
# ╟─16033d7b-e4aa-47f4-a79a-897f2599ae4c
# ╠═2b6cda5e-442c-4880-820b-a58d333ba307
# ╠═edf7dfc5-8325-418b-b3cd-23f568f061d1
# ╟─8d0125b6-4c83-43e2-8e69-86ac69d5e511
# ╠═97b45c4f-b527-4b07-99e6-b3c6c76c58f5
# ╟─353435cb-2f85-475f-93ba-eb1d170b2956
# ╟─abc20cde-f4bd-41ef-bcc1-60371e99829d
# ╠═05291062-9836-4f9b-a4b1-8a5e88ae17b3
# ╟─f5bab507-5a8f-4248-83f3-5556b2fbf771
# ╠═8962a0a1-f4ce-42ec-bb2e-5439f2580ff8
# ╟─0f2c526c-e3b9-403b-994e-a25caa721c6c
# ╟─fc4b6ada-756d-4ca1-b864-94a1de74373b
# ╠═cc3d3ffc-f7eb-422a-9427-54112aacf53c
# ╠═a215ab12-b332-4922-9011-c5b53610f5c3
# ╟─f12771e0-7898-4761-abfd-a567c2cf7b9f
# ╟─4106b5ca-8a2d-4ce3-8f97-99ee35235bf2
# ╠═3eb37f9e-74d9-492d-83a5-b84e6ac9278a
# ╠═25095605-224a-4888-bf29-75b663548c10
# ╠═715b3217-5779-4047-8c6a-8ce6162a0c00
# ╟─c4f287ea-47f3-437a-860d-874f6885d3fe
# ╟─076c20fa-b6fa-4754-a864-3643a2ba2abe
# ╠═ae2c0401-baa2-49b4-885b-6331a52c7069
# ╟─aa7de575-4f0c-4720-b659-e87a6ccf1b09
# ╠═3c7c4cf8-1689-4ddc-9258-efd80c610be1
# ╟─6107cc11-06b7-47d8-ab8d-662a3d5dc6bb
# ╠═c51b7a96-46d8-4852-903f-364cfe4d46c9
# ╠═9ee9c663-a236-4f0c-9ec2-b5278ffafcbe
# ╠═679d54d8-0334-4f4e-a24b-d0fbf0b98998
# ╠═0e0af689-56bc-4431-9daf-8c0106a94d81
# ╟─1b7be082-3e39-4497-ad0e-7ea0c96f19f1
# ╠═9d94fde8-135c-4ad4-9899-1704371ef0de
# ╟─b284a912-319d-416f-9093-5d3eb7233276
# ╠═ce099828-4870-4fbf-a754-d1a9c99969b8
# ╟─7e857775-2eac-4ce4-8137-75ccd9546107
# ╠═e5e9f0c9-a7fc-4820-a575-f22cc5b2775a
# ╟─53fe4346-0a75-4846-b99b-1d4b863a8dc8
# ╠═16555410-c22f-4726-af91-72c286a02897
# ╟─16a27687-ca90-4736-9079-3bed2b40a096
# ╟─8b196d4e-4367-4a22-8e7b-8be601517ef3
# ╠═ff385589-a034-4eab-be88-e74921164c67
# ╟─198cb590-2f8a-4df5-9d6f-efc6055cf4e0
# ╟─e2b13229-ab29-462e-bbd2-fefb3886a702
# ╠═00b5e789-e62e-440c-89c6-a26fd71633bc
# ╟─9c2c40d5-1938-4a3a-828d-d1f5fed19618
# ╠═a15260d5-3ef4-4a67-8512-810a2b76de49
# ╟─a73e8399-ba89-4ccb-b2a1-93c34d788509
# ╠═191fee5a-5e6e-4b91-b5f3-9919933e808e
# ╟─6fdac8e1-286b-46a3-a563-6fa5cca48c09
# ╠═9eba8bcb-7d99-4f78-a519-86a5dfd08035
# ╟─3ea37951-0d48-4110-bd80-ab9afed7ce37
# ╠═2f0ec430-440a-4b09-99ac-73f5558de07c
# ╟─5c6c4774-071d-42e3-b1b1-88cc55c8ce16
# ╟─7928b444-c7df-4750-9bdc-6711d7f981a5
# ╟─9b02f32d-aa48-4c7f-8984-d998735e4187
# ╟─1f971adb-c94e-477f-9d82-613344455915
# ╟─5c7f0e32-b888-4305-915f-b9d5b0a53520
# ╟─21d3f754-317a-42c3-a0ee-75a9f4a57468
# ╟─8d75120d-08af-43bb-95cd-a429b2888986
# ╟─340cc332-bbb1-4e04-a0db-f4fb49280c0e
# ╠═0d34b214-5e4d-4aba-af3f-180c82295254
# ╟─9a6b0c01-182b-4466-b204-fe8aa9316658
# ╟─053125c4-2c54-4509-b27a-c63cbe9cadb4
# ╟─f6370a67-5a05-4f5f-841b-c59171dc3ac6
# ╟─19ecb197-624b-49e1-9994-f7a35c7fce07
# ╠═ea0220db-574e-4ea4-b0b9-93eda5144d32
# ╟─009d9ffb-c37e-4bfd-8cb2-38c12bb5aee0
# ╠═fef1067d-91a5-4d13-ad11-4400c34fd5db
# ╟─cb7629fd-cb9a-4ff0-806a-5fdb1cb55b35
# ╟─438ccc1a-88fc-435b-9d5a-bc34312b5bcf
# ╟─c45ff159-9b32-4268-b52f-8430c4e02d61
# ╟─39e07c43-2d93-4745-8314-43ec69ad60ec
# ╟─6e7a8d1d-40b0-4319-a438-27eacf5fe754
# ╟─d5464c89-8185-4b63-b77c-e941a09dc95d
# ╠═1d7bd4c3-70f0-4eba-9536-570fa84148c5
# ╟─0537a88a-81c4-4523-9f18-7e3f9fc4e310
# ╠═42deb0ec-4beb-4dda-b206-65af56866577
# ╟─785eb34f-168b-4d2f-b08b-7a531cf66fde
# ╟─36016caa-c799-41bb-86f3-c57fc8739411
# ╟─8fc72595-166a-487e-9212-b6927f0f1876
# ╟─c994cbc8-e31c-41fa-b0c2-e935c6203f99
# ╟─eebfdc77-bbe0-407e-8559-66bbfe4f07da
# ╠═fd2b3bac-5bee-4f42-9ad7-c7ec15585a84
# ╟─83d2b40d-429e-40a4-b658-0b15c30b620b
# ╠═49c8663a-b113-4655-9ce1-1eaaf99221e6
# ╟─65e9ccd3-4ba4-4c7f-9720-65629bb02caf
# ╠═754c6392-a468-4dde-9ef3-eee177145237
# ╟─e33a6bb6-81a8-430e-a755-09e809a3a9c3
# ╟─a9447a6b-bf7c-4fa1-b501-d2957579446e
# ╟─eacc1145-c301-40a9-8dd9-33d3aeb060dc
# ╠═d4f1cd0b-a745-4463-a59a-68b8f7b4beee
# ╟─e93791f9-4ea9-488c-aff3-40965480ce42
# ╠═4eca1bb7-c43e-416c-b24d-488d3154b986
# ╠═e28c4c84-cfaa-4e9e-a358-015d3107dcb2
# ╟─1608b788-8ec5-44f4-9166-2d54e201a1c0
# ╠═a31f1a57-cb1a-4f43-98f2-5fc751f58c9f
# ╟─07956497-d26b-4d30-be45-bd70893062c5
# ╠═51e8eca7-bc01-4e02-b3f4-a9cad92dc3fa
# ╟─7d48f962-d80d-447b-b3b8-dc4e684cf2e4
# ╠═7bbfdb5a-2332-412f-8157-780df408bc74
# ╟─59311841-c40f-4046-82d6-ee0480252c31
# ╟─ff7c59b9-e2aa-4c75-96a2-b97eda9befd7
# ╠═21ad80f3-ffaf-4e85-814c-3a0609d7cc3b
# ╟─9269fb9d-13e6-49da-9a95-7f065c36b4d1
# ╠═24dab50f-d8cc-4129-b34f-2d9aa522191c
# ╟─9714bcc7-7dd4-471b-b781-6cf7786dd805
# ╠═b46d0235-5c0d-4439-926a-a9c10e541c48
# ╟─d02593d9-9f3a-488c-8fc3-b3fa2aff0f35
# ╠═ea7d1d69-12ff-4b58-aea7-d59f885d5e2b
# ╟─36d38f50-9038-420c-a3d0-c318a5ab2776
# ╠═4c1efec6-2499-43bc-ae85-873090fd6924
# ╟─a2fef07c-d41c-4263-bd69-011860beb824
# ╠═f2fa67c8-bb19-4b81-8d9d-5375d574bc39
# ╟─d2a44c70-00ef-41f7-86c2-4af3503df3e1
# ╠═dd0ebc7d-0b99-4d7c-b47b-8b44c34c9939
# ╟─60c8d206-528a-462b-ae3e-f684dc1a5163
# ╠═5e57095e-8962-4a0d-8993-5ddb5e2cf643
# ╟─5dd3e1d5-b408-488c-8e4f-391f061a024f
# ╠═3b2980a6-7b36-4353-9ff4-41a2bc5ab6ff
# ╟─1674f0ac-e3a8-4b3b-b398-e951add574a3
# ╟─f1b83d73-39e8-4f37-82b7-eec404341c6e
# ╠═e41a889a-8cd1-41de-a4fc-728c3ce38334
# ╟─d5648a74-65e2-4b14-8f0f-c4eea9e188c2
# ╠═f9caa6b4-2718-4fa7-bd85-5fc54d51b652
# ╟─bc229146-ef6e-49ba-8aa3-02fdad52a430
# ╠═182df020-a9e9-4d60-a579-d08d307e40f5
# ╟─db537cc3-09b0-4d6e-b5bb-2eb67877e258
# ╟─35944668-08b5-49ec-923f-d689f3f923b3
# ╠═668d011d-cade-4e97-b35b-21190ecca084
# ╟─38d3369a-8589-4d32-b36b-174aa07d183f
# ╠═35071e0f-b195-473a-a300-b50488d441e0
# ╟─dcc595d7-ae88-420e-8b5f-057e6d1ea0e0
# ╠═e19cf30e-03ef-42a6-88a4-04938faf9fe4
# ╟─34af0c6c-6f68-43a6-b899-5f61f39c277f
# ╠═a95e6d5b-9996-45db-b061-1cba033b1a6c
# ╠═5d456d2f-aa3c-47d0-b1c5-5131d08fc031
# ╠═2f36f77e-7739-4d60-b849-4d9adc48bf56
# ╟─4c4a3768-0afd-438f-b52c-a21c26463892
# ╟─7a7deb5e-3a6a-4cba-89e4-287f04cb2ae1
# ╟─58c2ed5e-4907-4358-8ee1-64b11b957be2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

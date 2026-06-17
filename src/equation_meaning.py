from cleaner import clean_text


text = "1. where \mathcal{M}_{n}=\{M_{x_{n}}\}_{x_{n}\in\mathcal{X}_{n}} denotes a POVM with |\mathcal{X}_{n}|>1 and 2. If Bob communicates classically, we will use the variable Y_{n} in place of V_{n} . Here, Y_{n} is obtained as the output of the measurement channel \mathcal{M}_{n}^{B^{n}\to Y_{n}}(\cdot) similar to ( [MENTION] ). When B^{n} is directly accessible to Charlie, e.g., via teleportation or quantum communication at a sufficiently large rate, then Bob may be identified with Charlie. In particular, an upper bound on the Stein’s exponent in this scenario will be an upper bound for the case when Bob communicates with Charlie. 3. One may obtain a multi-letter characterization of \theta(\epsilon,\rho_{AB},\tilde{\rho}_{AB}) in the limit \epsilon\downarrow 0^{+} by an adaptation of the proof of [ 21 , Theorem 1] to the quantum setting."

cleaned = clean_text(text)
print()
print(cleaned)


"""
1. need to get the one sentence above the equation still. 4 equation showed this is working
2. 

"""
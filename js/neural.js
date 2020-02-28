Math.sigmoid = function(x, derivative = false) {
    if (derivative) {
        return x * (1 - x);
    }
    return 1 / (1 + Math.exp(-x));
}

Math.angleToPoint = function(x, y, bearing, targetX, targetY) {
    let angleToTarget = Math.atan2(-targetY + y, targetX + x)
    let diff = bearing - angleToTarget
    return (diff + Math.PI * 2) % (Math.PI * 2)
}

const LOG = true
const LOG_COUNT = 1000

class Matrix {
    constructor(rows, cols, data = []) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;

        if (data == null || data.length == 0) {
            for (let i = 0; i < rows; i++) {
                this.data[i] = []
                for (let j = 0; j < cols; j++) {
                    this.data[i][j] = 0
                }
            }
        } else {
            if (data.length != rows || data[0].length != cols) {
                throw new Error("Invalid data dimensions")
            }
        }
    }

    map(cb) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = cb(i, j, this.data[i][j])
            }
        }
        return this
    }

    random() {
        this.map(() => {
            return Math.random() * 2 -1;
        })
    }

    static isSameLength(m1, m2) {
        if (m1.data.length == m2.data.length && m1.data[0].length == m2.data[0].length) return true;
        return false;
    }

    static add(m1, m2) {
        let m
        if (Matrix.isSameLength(m1, m2)) {
            m = new Matrix(m1.rows, m2.cols)
            for (let i = 0; i < m1.rows; i++) {
                for (let j = 0; j < m1.cols; j++) {
                    m.data[i][j] = m1.data[i][j] + m2.data[i][j]
                }
            }
        } else {
            throw new Error("The matrixes aren`t the same in size!")
        }
        return m
    }

    static fromArray(arr = []) {
        return new Matrix(1, arr.length, [arr]);
    }

    static transpose(m1) {
        let m = new Matrix(m1.cols, m1.rows)
        for (let i = 0; i < m1.rows; i++) {
            for (let j = 0; j < m1.cols; j++) {
                m.data[j][i] = m1.data[i][j]
            }
        }
        return m;
    }

    static subtract(m1, m2) {
        let m
        if (Matrix.isSameLength(m1, m2)) {
            m = new Matrix(m1.rows, m2.cols)
            for (let i = 0; i < m1.rows; i++) {
                for (let j = 0; j < m1.cols; j++) {
                    m.data[i][j] = m1.data[i][j] - m2.data[i][j]
                }
            }
        } else {
            throw new Error("The matrixes aren`t the same in size!")
        }
        return m
    }

    static multiply(m1, m2) {
        let m
        if (Matrix.isSameLength(m1, m2)) {
            m = new Matrix(m1.rows, m2.cols)
            for (let i = 0; i < m1.rows; i++) {
                for (let j = 0; j < m1.cols; j++) {
                    m.data[i][j] = m1.data[i][j] * m2.data[i][j]
                }
            }
        } else {
            throw new Error("The matrixes aren`t the same in size!")
        }
        return m
    }

    static dot(m1, m2) {
        if (m1.cols != m2.rows) {
            throw new Error("Not compatibles");
        }
        let m = new Matrix(m1.rows, m2.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                let sum = 0;
                for (let k = 0; k < m1.cols; k++) {
                    sum += m1.data[i][k] * m2.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }
}

class NeuralNetwork {
    constructor(inputs, hidden, outputs) {
        this.input = []
        this.hidden = []
        this.nInputs = inputs;
        this.nHidden = hidden;
        this.nOutputs = outputs;

        this.bias1 = new Matrix(1, hidden)
        this.bias2 = new Matrix(1, outputs)
        this.weights1 = new Matrix(inputs, hidden)
        this.weights2 = new Matrix(hidden, outputs)

        this.logCount = LOG_COUNT

        this.bias1.random()
        this.bias2.random()
        this.weights1.random()
        this.weights2.random()
    }

    feedForward(input) {
        this.input = Matrix.fromArray(input);

        this.hidden = Matrix.dot(this.input, this.weights1);
        this.hidden = Matrix.add(this.hidden, this.bias1);
        this.hidden.map((i, j, v) => Math.sigmoid(v));

        let output = Matrix.dot(this.hidden, this.weights2);
        output = Matrix.add(output, this.bias2);
        output.map((i, j, v) => Math.sigmoid(v));

        return output
    }

    train(input, target) {
        let output = this.feedForward(input)

        target = Matrix.fromArray(target)

        let outputError = Matrix.subtract(target, output)

        if (LOG) {
            if (this.logCount == LOG_COUNT) {
                console.log("Error: ", outputError.data[0][0])
            }
            this.logCount --
            if (this.logCount == 0) {
                this.logCount = LOG_COUNT
            }
        }

        let outputDerivs = output.map((i, j, v) => Math.sigmoid(v, true))
        let outputDelta = Matrix.multiply(outputError, outputDerivs)

        let weights2T = Matrix.transpose(this.weights2)
        let hiddenError = Matrix.dot(outputDelta, weights2T)

        let hiddenDerivs = this.hidden.map((i, j, v) => Math.sigmoid(v, true))
        let hiddenDelta = Matrix.multiply(hiddenError, hiddenDerivs)

        let hiddenT = Matrix.transpose(this.hidden)
        this.weights2 = Matrix.add(this.weights2, Matrix.dot(hiddenT, outputDelta)) 

        let inputT = Matrix.transpose(this.input)
        this.weights1 = Matrix.add(this.weights1, Matrix.dot(inputT, hiddenDelta))

        this.bias2 = Matrix.add(this.bias2, outputDelta)
        this.bias1 = Matrix.add(this.bias1, hiddenDelta)
    }
}
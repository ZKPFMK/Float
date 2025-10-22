package gpt2

import (
	"gnark-float/float"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

type GPT2Circuit struct {
	X [][]float.FloatVar
	W [][]float.FloatVar
	B [][]float.FloatVar
	Y [][]float.FloatVar
}

func (circuit *GPT2Circuit) Define(api frontend.API) error {
	ctx := float.NewContext(api, 0, 8, 23)
	xmn := mean(ctx, circuit.X[0])
	variance(ctx, circuit.X[0], xmn)
	return nil
}

func TestNormCircuit(t *testing.T) {
	assert := test.NewAssert(t)

	nRow, nCol := 1, 3
	tmpl := GPT2Circuit{
		X: make([][]float.FloatVar, nRow),
		W: make([][]float.FloatVar, nRow),
		B: make([][]float.FloatVar, nRow),
		Y: make([][]float.FloatVar, nRow),
	}
	for i := range tmpl.X {
		tmpl.X[i] = make([]float.FloatVar, nCol)
		tmpl.W[i] = make([]float.FloatVar, nCol)
		tmpl.B[i] = make([]float.FloatVar, nCol)
		tmpl.Y[i] = make([]float.FloatVar, nCol)
	}

	circuit := GPT2Circuit{
		X: make([][]float.FloatVar, nRow),
		W: make([][]float.FloatVar, nRow),
		B: make([][]float.FloatVar, nRow),
		Y: make([][]float.FloatVar, nRow),
	}

	for i := range circuit.X {
		circuit.X[i] = make([]float.FloatVar, nCol)
		circuit.W[i] = make([]float.FloatVar, nCol)
		circuit.B[i] = make([]float.FloatVar, nCol)
		circuit.Y[i] = make([]float.FloatVar, nCol)
	}

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			circuit.X[i][j] = float.NewF32Constant(2)
			circuit.W[i][j] = float.NewF32Constant(2)
			circuit.B[i][j] = float.NewF32Constant(3)
			circuit.Y[i][j] = float.NewF32Constant(4)
		}
	}

	for j := 0; j < nCol; j++ {
		circuit.X[0][j] = float.NewF32Constant(float32(j))
	}

	circuit.X[0][1] = float.NewF32Constant(float32(1))
	circuit.X[0][2] = float.NewF32Constant(float32(1))

	assert.NoError(test.IsSolved(&tmpl, &circuit, ecc.BN254.ScalarField()))

}

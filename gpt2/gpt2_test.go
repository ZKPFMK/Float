package gpt2

import (
	"gnark-float/float"
	"gnark-float/util"
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

	Eps float.FloatVar
}

func (circuit *GPT2Circuit) Define(api frontend.API) error {
	ctx := float.NewContext(api, 0, 8, 23)
	xmn := mean(ctx, circuit.X[0])
	vari := variance(ctx, circuit.X[0], xmn)
	x_hat := norm(ctx, circuit.X[0], xmn, vari, circuit.Eps)
	layerNorm(ctx, x_hat, circuit.W[0], circuit.B[0])
	return nil
}

func TestNormCircuit(t *testing.T) {
	assert := test.NewAssert(t)

	dir := "/home/dj/work/idea/pfllm/data/"
	input_path := dir + "input"
	bias11_path := dir + "block1_bias1"
	weight11_path := dir + "block1_weight1"

	input := util.Read2DFile(input_path)
	bias11 := util.Read1DFile(bias11_path)
	weight11 := util.Read1DFile(weight11_path)

	nRow, nCol := len(input), len(input[0])
	tmpl := GPT2Circuit{
		X:   make([][]float.FloatVar, nRow),
		W:   make([][]float.FloatVar, nRow),
		B:   make([][]float.FloatVar, nRow),
		Y:   make([][]float.FloatVar, nRow),
		Eps: float.NewF32ConstantFromFloat(float32(1e-5)),
	}
	for i := range tmpl.X {
		tmpl.X[i] = make([]float.FloatVar, nCol)
		tmpl.W[i] = make([]float.FloatVar, nCol)
		tmpl.B[i] = make([]float.FloatVar, nCol)
		tmpl.Y[i] = make([]float.FloatVar, nCol)
	}

	circuit := GPT2Circuit{
		X:   make([][]float.FloatVar, nRow),
		W:   make([][]float.FloatVar, nRow),
		B:   make([][]float.FloatVar, nRow),
		Y:   make([][]float.FloatVar, nRow),
		Eps: float.NewF32ConstantFromFloat(float32(1e-5)),
	}

	for i := range circuit.X {
		circuit.X[i] = make([]float.FloatVar, nCol)
		circuit.W[i] = make([]float.FloatVar, nCol)
		circuit.B[i] = make([]float.FloatVar, nCol)
		circuit.Y[i] = make([]float.FloatVar, nCol)
	}

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			circuit.X[i][j] = float.NewF32ConstantFromInt(uint64(input[i][j]))
			circuit.W[i][j] = float.NewF32ConstantFromInt(uint64(weight11[j]))
			circuit.B[i][j] = float.NewF32ConstantFromInt(uint64(bias11[j]))
			circuit.Y[i][j] = float.NewF32ConstantFromInt(uint64(input[i][j]))
		}
	}

	assert.NoError(test.IsSolved(&tmpl, &circuit, ecc.BN254.ScalarField()))

}

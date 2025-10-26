package float2

import (
	"bufio"
	"fmt"
	"gnark-float/hint"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

type F32UnaryCircuit struct {
	X  FloatVar
	Y  FloatVar
	op string
}

func (c *F32UnaryCircuit) Define(api frontend.API) error {
	ctx := NewContext(api, 0, 8, 23)
	_, _ = ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, c.X.Sign, c.X.Exponent, c.X.Mantissa, c.X.IsAbnormal)
	ctx.Api.AssertIsBoolean(c.X.Sign)
	ctx.Api.AssertIsBoolean(c.X.IsAbnormal)

	ctx.Api.AssertIsBoolean(c.Y.Sign)
	ctx.Api.AssertIsBoolean(c.Y.IsAbnormal)
	// ctx.AssertIsEqual(reflect.ValueOf(&ctx).MethodByName(c.op).Call([]reflect.Value{reflect.ValueOf(c.X)})[0].Interface().(FloatVar), c.Y)
	return nil
}

type F32BinaryCircuit struct {
	X  FloatVar
	Y  FloatVar
	Z  FloatVar
	op string
}

func (c *F32BinaryCircuit) Define(api frontend.API) error {
	ctx := NewContext(api, 0, 8, 23)
	ctx.AssertIsEqual(reflect.ValueOf(&ctx).MethodByName(c.op).Call([]reflect.Value{reflect.ValueOf(c.X), reflect.ValueOf(c.Y)})[0].Interface().(FloatVar), c.Z)
	return nil
}

func TestF32UnaryCircuit(t *testing.T) {
	assert := test.NewAssert(t)

	// ops := []string{"Sqrt", "Trunc", "Floor", "Ceil"}
	ops := []string{"Sqrt"}

	for _, op := range ops {
		path, _ := filepath.Abs(fmt.Sprintf("../data/f32/%s", strings.ToLower(op)))
		file, _ := os.Open(path)
		defer file.Close()

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			data := strings.Fields(scanner.Text())
			a, _ := new(big.Int).SetString(data[0], 16)
			b, _ := new(big.Int).SetString(data[1], 16)
			assert.NoError(
				test.IsSolved(
					&F32UnaryCircuit{X: NewF32ConstantFromFloat(0.0), Y: NewF32ConstantFromFloat(0.0), op: op},
					&F32UnaryCircuit{X: NewF32ConstantFromInt(uint32(a.Uint64())), Y: NewF32ConstantFromInt(uint32(b.Uint64())), op: op},
					ecc.BN254.ScalarField(),
				))
		}
	}
}

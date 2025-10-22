package gpt2

import (
	"fmt"
	"gnark-float/float"
	"gnark-float/hint"
)

// type NormGadget struct {
// 	api frontend.API
// 	row uint
// 	col uint
// 	eps float32
// }

// func NewNormGadget(api frontend.API, row uint, col uint, eps float32) *NormGadget {
// 	return &NormGadget{api, row, col, eps}
// }

func mean(ctx float.Context, v []float.FloatVar) float.FloatVar {
	var sum float.FloatVar = float.NewF32Constant(0)
	for i := 0; i < len(v); i++ {
		sum = ctx.Add(sum, v[i])
	}
	ret := ctx.Mul(sum, ctx.NewF32Constant(1.0/float32(len(v))))
	return ret
}

func variance(ctx float.Context, v []float.FloatVar, mn float.FloatVar) float.FloatVar {
	var sum float.FloatVar = float.NewF32Constant(0)
	for i := 0; i < len(v); i++ {
		diff := ctx.Sub(v[i], mn)
		sum = ctx.Add(sum, ctx.Mul(diff, diff))
		fmt.Printf("diff:")
		ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, diff.Sign, diff.Exponent, diff.Mantissa, diff.IsAbnormal)
	}
	inv := ctx.NewF32Constant(1.0 / float32(len(v)))
	ret := ctx.Mul(sum, inv)

	fmt.Printf("inv:")
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, inv.Sign, inv.Exponent, inv.Mantissa, inv.IsAbnormal)

	fmt.Printf("sum:")
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, sum.Sign, sum.Exponent, sum.Mantissa, sum.IsAbnormal)

	fmt.Printf("mean:")
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, mn.Sign, mn.Exponent, mn.Mantissa, mn.IsAbnormal)

	fmt.Printf("variance:")
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, ret.Sign, ret.Exponent, ret.Mantissa, ret.IsAbnormal)
	return ret
}

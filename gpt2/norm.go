package gpt2

import (
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
	var sum float.FloatVar = float.NewF32ConstantFromFloat(0.0)
	for i := 0; i < len(v); i++ {
		sum = ctx.Add(sum, v[i])
	}
	ret := ctx.Mul(sum, ctx.NewF32Constant(1.0/float32(len(v))))
	return ret
}

func variance(ctx float.Context, v []float.FloatVar, mean float.FloatVar) float.FloatVar {
	var sum float.FloatVar = float.NewF32ConstantFromFloat(0.0)
	for i := 0; i < len(v); i++ {
		diff := ctx.Sub(v[i], mean)
		sum = ctx.Add(sum, ctx.Mul(diff, diff))
	}
	inv := ctx.NewF32Constant(1.0 / float32(len(v)))
	ret := ctx.Mul(sum, inv)
	return ret
}

func norm(ctx float.Context, v []float.FloatVar, mean float.FloatVar, variance float.FloatVar, eps float.FloatVar) []float.FloatVar {
	ret := make([]float.FloatVar, len(v))
	for i := 0; i < len(v); i++ {
		diff := ctx.Sub(v[i], mean) //这里可以优化, 在计算方差时已经有了diff
		ret[i] = ctx.Div(diff, ctx.Sqrt(ctx.Add(variance, eps)))
	}
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, mean.Sign, mean.Exponent, mean.Mantissa, mean.IsAbnormal)
	ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, variance.Sign, variance.Exponent, variance.Mantissa, variance.IsAbnormal)
	return ret
}

// 10552398 119
func layerNorm(ctx float.Context, v []float.FloatVar, w []float.FloatVar, b []float.FloatVar) []float.FloatVar {
	ret := make([]float.FloatVar, len(v))
	for i := 0; i < len(v); i++ {
		ret[i] = ctx.Add(ctx.Mul(v[i], w[i]), b[i])
	}

	for i := 0; i < 5; i++ {
		ctx.Api.Compiler().NewHint(hint.PrintHint32, 1, ret[i].Sign, ret[i].Exponent, ret[i].Mantissa, ret[i].IsAbnormal)
	}
	return ret
}

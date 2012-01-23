percentPositive <- function(ionoGram, cutoff)
{
	.Call("percentPositive", ionoGram, cutoff)
}

sumFractionalPart <- function(ionogram)
{
	.Call("sumFractionalPart", ionogram)
}

fitNormals <- function(ppf, ssq)
{
	.Call("fitNormals", ppf, ssq)
}

distanceFromMean <- function(mean, sigma, x)
{
	.Call("distanceFromMean", mean, sigma, x)
}


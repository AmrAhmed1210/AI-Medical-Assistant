using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace MedicalAssistant.Web.Filters;

public class SwaggerFileUploadFilter : IOperationFilter
{
    public void Apply(OpenApiOperation operation, OperationFilterContext context)
    {
        var formFileParameters = context.MethodInfo.GetParameters()
            .Where(p => p.ParameterType == typeof(IFormFile) || p.ParameterType == typeof(IFormFile[]))
            .ToList();

        if (!formFileParameters.Any())
            return;

        operation.RequestBody = new OpenApiRequestBody
        {
            Content = new Dictionary<string, OpenApiMediaType>
            {
                ["multipart/form-data"] = new OpenApiMediaType
                {
                    Schema = new OpenApiSchema
                    {
                        Type = "object",
                        Properties = formFileParameters.ToDictionary(
                            p => p.Name!,
                            p => new OpenApiSchema
                            {
                                Type = "string",
                                Format = p.ParameterType == typeof(IFormFile[]) ? "binary" : "binary"
                            }
                        ),
                        Required = formFileParameters
                            .Where(p => !p.IsOptional)
                            .Select(p => p.Name!)
                            .ToHashSet()
                    }
                }
            }
        };

        // Remove the auto-generated parameters for IFormFile
        foreach (var param in formFileParameters)
        {
            var toRemove = operation.Parameters
                .FirstOrDefault(p => p.Name == param.Name);
            if (toRemove != null)
                operation.Parameters.Remove(toRemove);
        }
    }
}

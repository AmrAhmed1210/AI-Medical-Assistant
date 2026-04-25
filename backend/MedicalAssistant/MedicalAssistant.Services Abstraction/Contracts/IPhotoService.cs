using Microsoft.AspNetCore.Http;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IPhotoService
{
    Task<string> UploadPhotoAsync(IFormFile file);
    Task<string> UploadFileAsync(IFormFile file); // For CVs
    Task<bool> DeletePhotoAsync(string publicId);
}

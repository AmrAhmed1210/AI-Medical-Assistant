using CloudinaryDotNet;
using CloudinaryDotNet.Actions;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.Settings;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace MedicalAssistant.Services.Services;

public class PhotoService : IPhotoService
{
    private readonly Cloudinary _cloudinary;
    private readonly ILogger<PhotoService> _logger;

    public PhotoService(IOptions<CloudinarySettings> config, ILogger<PhotoService> logger)
    {
        _logger = logger;

        var acc = new Account(
            config.Value.CloudName,
            config.Value.ApiKey,
            config.Value.ApiSecret
        );

        _cloudinary = new Cloudinary(acc);
        _cloudinary.Api.Timeout = 120000; // 120 seconds timeout
        Console.WriteLine($"[DEBUG] Using Cloudinary API Key starting with: {config.Value.ApiKey?.Substring(0, 4)}...");
    }

    public async Task<string> UploadPhotoAsync(IFormFile file)
    {
        if (file.Length > 0)
        {
            // Read the entire file into a MemoryStream first to avoid I/O abort
            // when the HTTP request stream is disposed before Cloudinary finishes
            var memoryStream = new MemoryStream();
            await file.CopyToAsync(memoryStream);
            memoryStream.Position = 0;

            var uploadParams = new ImageUploadParams
            {
                File = new FileDescription(file.FileName, memoryStream),
                Transformation = new Transformation().Height(500).Width(500).Crop("fill").Gravity("face"),
                Folder = "medbook-photos"
            };

            try 
            {
                var uploadResult = await _cloudinary.UploadAsync(uploadParams);

                if (uploadResult.Error != null)
                {
                    _logger.LogError("Cloudinary Upload Error: {Message}", uploadResult.Error.Message);
                    throw new Exception($"Cloudinary Error: {uploadResult.Error.Message}");
                }

                return uploadResult.SecureUrl.ToString();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exception during Cloudinary photo upload");
                throw;
            }
        }

        return string.Empty;
    }

    public async Task<string> UploadFileAsync(IFormFile file)
    {
        if (file.Length > 0)
        {
            // Read the entire file into a MemoryStream first to avoid I/O abort
            var memoryStream = new MemoryStream();
            await file.CopyToAsync(memoryStream);
            memoryStream.Position = 0;

            var uploadParams = new RawUploadParams
            {
                File = new FileDescription(file.FileName, memoryStream),
                Folder = "medbook-docs"
            };

            var uploadResult = await _cloudinary.UploadAsync(uploadParams);

            if (uploadResult.Error != null)
            {
                throw new Exception(uploadResult.Error.Message);
            }

            return uploadResult.SecureUrl.ToString();
        }

        return string.Empty;
    }

    public async Task<bool> DeletePhotoAsync(string publicId)
    {
        var deleteParams = new DeletionParams(publicId);

        var result = await _cloudinary.DestroyAsync(deleteParams);

        return result.Result == "ok";
    }
}

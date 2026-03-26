using MedicalAssistant.Shared.DTOs.Auth;

namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface IAuthService
{
    Task<AuthResponseDto?> LoginAsync(LoginDto loginDto);
    Task<bool> RegisterAdminAsync(RegisterDto registerDto);
}
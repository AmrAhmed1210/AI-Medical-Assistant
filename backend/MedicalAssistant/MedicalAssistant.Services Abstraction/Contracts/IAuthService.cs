using MedicalAssistant.Shared.DTOs.AuthDTOs;

namespace MedicalAssistant.Services_Abstraction.Contracts
{
    public interface IAuthService
    {
        Task<AuthResponseDto> RegisterAsync(RegisterDto dto);
        Task<AuthResponseDto> LoginAsync(LoginDto dto);
    }
}
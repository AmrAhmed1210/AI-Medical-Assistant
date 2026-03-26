using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Auth;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

namespace MedicalAssistant.Services.Services;

public class AuthService : IAuthService
{
    private readonly IUnitOfWork _unitOfWork;
    private readonly IConfiguration _config;

    public AuthService(IUnitOfWork unitOfWork, IConfiguration config)
    {
        _unitOfWork = unitOfWork;
        _config = config;
    }

    public async Task<AuthResponseDto?> LoginAsync(LoginDto loginDto)
    {
        var users = await _unitOfWork.Repository<User>().GetAllAsync();
        
        var user = users.FirstOrDefault(u => 
            u.Email.Trim().ToLower() == loginDto.Email.Trim().ToLower());

        if (user == null) 
            throw new Exception("DEBUG: Email not found in DB!");

        if (loginDto.Password.Trim() != user.PasswordHash.Trim())
        {
            throw new Exception($"DEBUG: Password mismatch! Provided: '{loginDto.Password}' | In DB: '{user.PasswordHash}'");
        }

        var token = GenerateJwtToken(user);

        return new AuthResponseDto
        {
            AccessToken = token,
            RefreshToken = Guid.NewGuid().ToString(),
            User = new UserDto 
            { 
                Id = user.Id, 
                FullName = user.FullName, 
                Email = user.Email, 
                Role = user.Role 
            }
        };
    }

    public async Task<bool> RegisterAdminAsync(RegisterDto registerDto)
    {
        var user = new User
        {
            FullName = registerDto.FullName,
            Email = registerDto.Email,
            PasswordHash = registerDto.Password,
            Role = "Admin",
            IsActive = true
        };

        await _unitOfWork.Repository<User>().AddAsync(user);
        return await _unitOfWork.SaveChangesAsync() > 0; 
    }

    private string GenerateJwtToken(User user)
    {
        try 
        {
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
                new Claim(ClaimTypes.Email, user.Email),
                new Claim(ClaimTypes.Role, user.Role),
                new Claim("FullName", user.FullName)
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["JWT:Key"] ?? "YourSuperSecretKeyShouldBeAtLeast32CharsLong!"));
            var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

            var token = new JwtSecurityToken(
                issuer: _config["JWT:Issuer"],
                audience: _config["JWT:Audience"],
                claims: claims,
                expires: DateTime.Now.AddDays(1),
                signingCredentials: creds
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"JWT Generation Error: {ex.Message}");
            throw new Exception($"Failed to generate Token: {ex.Message}");
        }
    }
}
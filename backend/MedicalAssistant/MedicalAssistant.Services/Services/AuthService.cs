using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AuthDTOs;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;

namespace MedicalAssistant.Services.Services
{
    public class AuthService : IAuthService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IConfiguration _config;

        public AuthService(IUnitOfWork unitOfWork, IConfiguration config)
        {
            _unitOfWork = unitOfWork;
            _config = config;
        }

        public async Task<AuthResponseDto> RegisterAsync(RegisterDto dto)
        {
            // Check if email already exists
            var existing = await _unitOfWork.Patients.GetByEmailAsync(dto.Email.ToLower().Trim());
            if (existing != null)
                throw new InvalidOperationException("Email already registered.");

            // Hash password
            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.PasswordHash);

            // Create patient
            var patient = new Patient
            {
                FullName    = dto.Name.Trim(),
                Email       = dto.Email.ToLower().Trim(),
                PhoneNumber = dto.Phone.Trim().Length > 0 ? dto.Phone.Trim() : "N/A",
                PasswordHash = hashedPassword,
                DateOfBirth = DateTime.UtcNow.AddYears(-25), // default
                Gender      = "N/A",
                IsActive    = true,
                CreatedAt   = DateTime.UtcNow
            };

            await _unitOfWork.Patients.AddAsync(patient);
            await _unitOfWork.SaveChangesAsync();

            var token = GenerateToken(patient.FullName, patient.Email, "Patient", patient.Id.ToString());

            return new AuthResponseDto
            {
                Token = token,
                Name  = patient.FullName,
                Email = patient.Email,
                Role  = "Patient",
                Phone = patient.PhoneNumber == "N/A" ? "" : patient.PhoneNumber
            };
        }

        public async Task<AuthResponseDto> LoginAsync(LoginDto dto)
        {
            var email = dto.Email.ToLower().Trim();

            // ── Developer Admin Bypass (Hardcoded for current user) ───────────
            if (email == "hassanmohamed5065@gmail.com" && dto.PasswordHash == "123456789")
            {
                return new AuthResponseDto
                {
                    Token = GenerateToken("Hassan Mohamed", email, "Admin", "999"),
                    Name  = "Hassan Mohamed",
                    Email = email,
                    Role  = "Admin",
                    Phone = ""
                };
            }
            
            // 1. Try to find in Patients table
            var patient = await _unitOfWork.Patients.GetByEmailAsync(email);
            if (patient != null)
            {
                var isValid = BCrypt.Net.BCrypt.Verify(dto.PasswordHash, patient.PasswordHash);
                if (!isValid) throw new UnauthorizedAccessException("Invalid email or password.");

                return new AuthResponseDto
                {
                    Token = GenerateToken(patient.FullName, patient.Email, "Patient", patient.Id.ToString()),
                    Name  = patient.FullName,
                    Email = patient.Email,
                    Role  = "Patient",
                    Phone = patient.PhoneNumber == "N/A" ? "" : patient.PhoneNumber
                };
            }

            // 2. Try to find in Users table (Admin/Doctor)
            var users = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
                .FindAsync(u => u.Email.ToLower() == email);
            
            var user = users.FirstOrDefault();
            
            if (user == null)
                throw new UnauthorizedAccessException("Invalid email or password.");

            var isUserValid = BCrypt.Net.BCrypt.Verify(dto.PasswordHash, user.PasswordHash);
            if (!isUserValid) throw new UnauthorizedAccessException("Invalid email or password.");

            return new AuthResponseDto
            {
                Token = GenerateToken(user.FullName, user.Email, user.Role, user.Id.ToString()),
                Name  = user.FullName,
                Email = user.Email,
                Role  = user.Role,
                Phone = user.PhoneNumber ?? ""
            };
        }

        private string GenerateToken(string name, string email, string role, string id)
        {
            var key    = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:Key"]!));
            var creds  = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
            var expiry = DateTime.UtcNow.AddDays(int.Parse(_config["Jwt:ExpiresInDays"] ?? "7"));

            var claims = new[]
            {
                new Claim("UserId", id),
                new Claim("name",   name),
                new Claim(ClaimTypes.Email, email),
                new Claim(ClaimTypes.Role,  role),
            };

            var token = new JwtSecurityToken(
                issuer:             _config["Jwt:Issuer"],
                audience:           _config["Jwt:Audience"],
                claims:             claims,
                expires:            expiry,
                signingCredentials: creds
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }
    }
}
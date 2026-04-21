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
        private readonly INotificationService _notificationService;

        public AuthService(IUnitOfWork unitOfWork, IConfiguration config, INotificationService notificationService)
        {
            _unitOfWork = unitOfWork;
            _config = config;
            _notificationService = notificationService;
        }

        public async Task<AuthResponseDto> RegisterAsync(RegisterDto dto)
        {
            var email = dto.Email.ToLower().Trim();

            // Check if email already exists
            var existing = await _unitOfWork.Patients.GetByEmailAsync(email);
            if (existing != null)
                throw new InvalidOperationException("Email already registered.");

            var existingUser = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
                .FindAsync(u => u.Email.ToLower() == email))
                .FirstOrDefault();

            if (existingUser != null)
            {
                if (existingUser.IsDeleted)
                    throw new InvalidOperationException("This account no longer exists.");

                throw new InvalidOperationException("Email already registered.");
            }

            // Hash password
            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);

            // Create patient
            var patient = new Patient
            {
                FullName    = dto.FullName.Trim(),
                Email       = email,
                PhoneNumber = dto.PhoneNumber?.Trim().Length > 0 ? dto.PhoneNumber.Trim() : "N/A",
                PasswordHash = hashedPassword,
                DateOfBirth = dto.DateOfBirth ?? DateTime.UtcNow.AddYears(-25),
                Gender      = "N/A",
                IsActive    = true,
                CreatedAt   = DateTime.UtcNow
            };

            await _unitOfWork.Patients.AddAsync(patient);
            await _unitOfWork.SaveChangesAsync();

            // Notify admin panel about new user registration
            await _notificationService.NotifyNewUserRegistered(
                patient.Id, patient.FullName, patient.Email, "Patient");

            var token = GenerateToken(patient.FullName, patient.Email, "Patient", patient.Id.ToString());

            return new AuthResponseDto
            {
                AccessToken = token,
                RefreshToken = "",
                ExpiresIn = 86400,
                User = new UserDto
                {
                    Id = patient.Id,
                    FullName = patient.FullName,
                    Email = patient.Email,
                    Role = "Patient"
                }
            };
        }

        public async Task<AuthResponseDto> LoginAsync(LoginDto dto)
        {
            var email = dto.Email.ToLower().Trim();

            // ── Developer Admin Bypass (Hardcoded for current user) ───────────
            if (email == "hassanmohamed5065@gmail.com" && dto.Password == "123456789")
            {
                return new AuthResponseDto
                {
                    AccessToken = GenerateToken("Hassan Mohamed", email, "Admin", "999"),
                    RefreshToken = "",
                    ExpiresIn = 86400,
                    User = new UserDto
                    {
                        Id = 999,
                        FullName = "Hassan Mohamed",
                        Email = email,
                        Role = "Admin"
                    }
                };
            }
            
            // 1. Try to find in Patients table
            var patient = await _unitOfWork.Patients.GetByEmailAsync(email);
            if (patient != null)
            {
                if (!patient.IsActive)
                    throw new UnauthorizedAccessException("Your account has been deactivated. Please contact administration at support@yourapp.com.");

                var isValid = BCrypt.Net.BCrypt.Verify(dto.Password, patient.PasswordHash);
                if (!isValid) throw new UnauthorizedAccessException("Invalid email or password.");

                return new AuthResponseDto
                {
                    AccessToken = GenerateToken(patient.FullName, patient.Email, "Patient", patient.Id.ToString()),
                    RefreshToken = "",
                    ExpiresIn = 86400,
                    User = new UserDto
                    {
                        Id = patient.Id,
                        FullName = patient.FullName,
                        Email = patient.Email,
                        Role = "Patient"
                    }
                };
            }

            // 2. Try to find in Users table (Admin/Doctor)
            var users = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
                .FindAsync(u => u.Email.ToLower() == email);
            
            var user = users.FirstOrDefault();
            
            if (user == null)
                throw new UnauthorizedAccessException("Invalid email or password.");

            // Check if account is deleted or inactive
            if (user.IsDeleted)
                throw new UnauthorizedAccessException("This account no longer exists.");
            
            if (!user.IsActive)
                throw new UnauthorizedAccessException("Your account has been deactivated. Please contact administration at support@yourapp.com.");

            var isUserValid = BCrypt.Net.BCrypt.Verify(dto.Password, user.PasswordHash);
            if (!isUserValid) throw new UnauthorizedAccessException("Invalid email or password.");

            string? doctorId = null;
            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doc = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Doctor>()
                    .FindAsync(d => d.UserId == user.Id);
                doctorId = doc.FirstOrDefault()?.Id.ToString();
            }

            return new AuthResponseDto
            {
                AccessToken = GenerateToken(user.FullName, user.Email, user.Role, user.Id.ToString(), doctorId),
                RefreshToken = "",
                ExpiresIn = 86400,
                User = new UserDto
                {
                    Id = user.Id,
                    FullName = user.FullName,
                    Email = user.Email,
                    Role = user.Role
                }
            };
        }

        private string GenerateToken(string name, string email, string role, string id, string? doctorId = null)
        {
            var key    = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:Key"]!));
            var creds  = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
            var expiry = DateTime.UtcNow.AddDays(int.Parse(_config["Jwt:ExpiresInDays"] ?? "7"));

            var claimsList = new List<Claim>
            {
                new Claim("UserId", id),
                new Claim("PatientId", id),  // For appointment controller compatibility
                new Claim("name",   name),
                new Claim(ClaimTypes.Email, email),
                new Claim(ClaimTypes.Role,  role),
            };

            if (!string.IsNullOrEmpty(doctorId))
            {
                claimsList.Add(new Claim("DoctorId", doctorId));
            }

            var token = new JwtSecurityToken(
                issuer:             _config["Jwt:Issuer"],
                audience:           _config["Jwt:Audience"],
                claims:             claimsList,
                expires:            expiry,
                signingCredentials: creds
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }
    }
}

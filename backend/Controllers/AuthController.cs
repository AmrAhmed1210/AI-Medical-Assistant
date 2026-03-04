using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using backend.Data;
using backend.Models;
using backend.DTOs;
using backend.Services;
using backend.Utils;

namespace backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ITokenService _tokenService;

        public AuthController(ApplicationDbContext context, ITokenService tokenService)
        {
            _context = context;
            _tokenService = tokenService;
        }

        // --- دالة التسجيل الجديدة ---
        [HttpPost("register")]
        public async Task<ActionResult<AuthResponse>> Register(UserDto registerDto)
        {
            // 1. التأكد إن الإيميل مش مستخدم قبل كدة
            if (await _context.Users.AnyAsync(x => x.Email == registerDto.Email.ToLower()))
                return BadRequest("Email is already taken");

            // 2. إنشاء كائن المستخدم الجديد وتشفير الباسورد
            var user = new User
            {
                Name = registerDto.Name,
                Email = registerDto.Email.ToLower(),
                PasswordHash = PasswordHasher.HashPassword(registerDto.PasswordHash), // تشفير الباسورد
                Role = registerDto.Role ?? "Patient", // القيمة الافتراضية
                CreatedAt = DateTime.Now
            };

            // 3. الحفظ في قاعدة البيانات
            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            // 4. إرجاع بيانات المستخدم مع الـ Token
            return new AuthResponse
            {
                Token = _tokenService.CreateToken(user),
                Name = user.Name,
                Role = user.Role
            };
        }

        [HttpPost("login")]
        public async Task<ActionResult<AuthResponse>> Login(LoginRequest loginRequest)
        {
            var user = await _context.Users.FirstOrDefaultAsync(x => x.Email == loginRequest.Email.ToLower());

            if (user == null) return Unauthorized("Invalid Email");

            var result = PasswordHasher.VerifyPassword(loginRequest.Password, user.PasswordHash);
            if (!result) return Unauthorized("Invalid Password");

            return new AuthResponse
            {
                Token = _tokenService.CreateToken(user),
                Name = user.Name,
                Role = user.Role
            };
        }
    }
}
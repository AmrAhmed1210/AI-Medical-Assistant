using backend.Models;

namespace backend.Services
{
    public interface ITokenService
    {
        string CreateToken(User user);
    }
}